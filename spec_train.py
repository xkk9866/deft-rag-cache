"""Margin distillation: fine-tune the speculator g so that its teacher-forced margin on the cached tokens matches the
reader's measured margin.

Every training example is a hop edit K* -> K' from the training streams (oracle replay edits, single-document
counterfactuals and synthetic edits; the same examples the MTR was trained on) plus, optionally, the trusted states
themselves (K' = K*).  The input is the reader's own prompt on K' followed by the reader's cached tokens y; the output
is the speculator's signed margin  m_g(y; K') = min_i [z_{y_i} - max_{v != y_i} z_v]  at the answer positions.  The
loss is the censored logistic likelihood of the reader's measured margin m_G(y; K') (exact, >= c or <= c), with one
learned scale.  No head is added: the LM's own logits are the output, so the distilled model remains a reader whose
margins are directly comparable with the reader it speculates for.  LoRA keeps the memory small.
Output: runs/models/spec_<spec><tag>/ (adapter + scale.json)
"""
import argparse, math, random, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *
from mcert import MarginBook, MCAP
from deltar import margin_examples
from spec import SPEC_PATHS, hop_prompt, spec_gaps_train

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="q06")
ap.add_argument("--wls", default="wiki_cf,hotpot,wiki_t")
ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--micro", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora", type=int, default=32, help="LoRA rank (0 = full fine-tuning)")
ap.add_argument("--maxlen", type=int, default=1536)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="_d0")
ap.add_argument("--trusted", action="store_true", help="add the trusted states (K' = K*) with their margins")
ap.add_argument("--binary", action="store_true", help="ablation: binary defeat likelihood instead of the margin likelihood")
ap.add_argument("--judge", action="store_true", help="alternative: fine-tune the same model as a yes/no judge of the cached answer (binary labels)")
ap.add_argument("--nosynth", action="store_true")
ap.add_argument("--limit", type=int, default=0, help="debug: cap the number of examples")
args = ap.parse_args()
torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)

path = SPEC_PATHS.get(args.spec, args.spec)
tok = AutoTokenizer.from_pretrained(path)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
if args.judge:
    from spec import Speculator
    _yes = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for v in ("yes", "Yes", " yes", " Yes")})
    _no = sorted({tok(v, add_special_tokens=False)["input_ids"][0] for v in ("no", "No", " no", " No")})

    def judge_ids(T, q, rows, a):
        docs = "\n".join(f"[{i + 1}] {T[r]}" for i, r in enumerate(rows))
        msgs = [{"role": "user", "content": Speculator.JUDGE.format(docs=docs, q=q, a=a)}]
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False), add_special_tokens=False)["input_ids"]

# ------------------------------------------------------------------ data
data = []
for wl in args.wls.split(","):
    exs, T = margin_examples(wl, "train", synth=not args.nosynth)
    MB = MarginBook(wl, "train")
    n0 = len(data)
    pm = {}
    def pids(q, rows):
        k = (q, rows)
        if k not in pm:
            pm[k] = tok(hop_prompt(tok, T, q, rows), add_special_tokens=False)["input_ids"]
        return pm[k]
    for e in exs:
        K, K2 = tuple(x for _, x in e["K"]), tuple(x for _, x in e["K2"])
        y = MB.tokens(e["subq"], K)
        if not y:
            continue
        if args.judge:                                   # judge: prompt with the cached answer, label = not defeated
            data.append((judge_ids(T, e["subq"], K2, e["ans"]), [], 0.0, 0, int(e["y"])))
            continue
        data.append((pids(e["subq"], K2), list(y), float(e["m2"]), int(e["cen"]), int(e["y"])))
    if args.trusted:
        seen = set((e["subq"], tuple(x for _, x in e["K"])) for e in exs)
        for (q, rows), st in MB.S.items():
            if (q, rows) in seen and st["tok"]:
                m = MB.mu(q, rows)
                if args.judge:
                    data.append((judge_ids(T, q, rows, st["ans"]), [], 0.0, 0, 0))
                else:
                    data.append((pids(q, rows), list(st["tok"]), float(min(m, MCAP)), 1 if m >= MCAP - 1e-3 else 0, 0))
    print(wl, "examples", len(data) - n0, flush=True)
if args.limit:
    random.shuffle(data); data = data[: args.limit]
data = [d for d in data if len(d[0]) + len(d[1]) <= args.maxlen]
print("train examples", len(data), "defeating", sum(d[4] for d in data), "mean prompt tokens",
      float(np.mean([len(d[0]) for d in data])), flush=True)

# ------------------------------------------------------------------ model
lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
lm.config.use_cache = False
if args.lora:
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=args.lora, lora_alpha=2 * args.lora, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    lm = get_peft_model(lm, cfg)
    lm.print_trainable_parameters()
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm.enable_input_require_grads()
    params = [p for p in lm.parameters() if p.requires_grad]
    inner = lm.base_model.model
else:
    for n_, p_ in lm.named_parameters():
        if "embed_tokens" in n_ or "lm_head" in n_:
            p_.requires_grad_(False)
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    params = [p for p in lm.parameters() if p.requires_grad]
    inner = lm
log_s = nn.Parameter(torch.tensor(math.log(2.0), device="cuda"))
import bitsandbytes as bnb
opt = bnb.optim.AdamW8bit([{"params": params, "lr": args.lr}, {"params": [log_s], "lr": 1e-2}], weight_decay=0.0)
steps = int(args.epochs * math.ceil(len(data) / args.bs))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * steps))) * max(0.0, (steps - s) / steps))


def batch_tensors(B):
    T = max(len(p) + len(y) for p, y, *_ in B)
    ids = torch.full((len(B), T), PAD, dtype=torch.long); att = torch.zeros((len(B), T), dtype=torch.long)
    pos, tgt = [], []
    for b, (p, y, *_) in enumerate(B):
        s = p + y
        ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
        pos.append([len(p) - 1 + i for i in range(len(y))] if y else [len(p) - 1]); tgt.append(torch.tensor(y, device="cuda"))
    return ids.cuda(), att.cuda(), pos, tgt


def judge_scores_train(lm_inner, ids, att, pos):
    """yes/no judge: log P(yes) - log P(no) at the last prompt position"""
    base = lm_inner.get_decoder() if hasattr(lm_inner, "get_decoder") else lm_inner.model
    h = base(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    last = torch.tensor([p[0] for p in pos], device=ids.device)
    z = torch.log_softmax(lm_inner.lm_head(h[torch.arange(len(pos), device=ids.device), last]).float(), -1)
    return torch.logsumexp(z[:, _yes], -1) - torch.logsumexp(z[:, _no], -1)


def loss_fn(m_hat, B):
    s = log_s.exp()
    m2 = torch.tensor([b[2] for b in B], device="cuda"); cen = torch.tensor([b[3] for b in B], device="cuda")
    if args.binary or args.judge:
        yb = torch.tensor([float(b[4]) for b in B], device="cuda")
        return F.binary_cross_entropy_with_logits(-m_hat / s, yb)
    u = (m2 - m_hat) / s
    exact = -F.logsigmoid(u) - F.logsigmoid(-u) + torch.log(s)
    right = -F.logsigmoid(-u); left = -F.logsigmoid(u)
    return torch.where(cen == 0, exact, torch.where(cen > 0, right, left)).mean()


lm.train()
step, t0 = 0, time.time()
n_ep = math.ceil(args.epochs)
for ep in range(n_ep):
    random.shuffle(data)
    # length-bucketed batches: sort within chunks so that micro-batches have similar lengths
    chunks = [sorted(data[i:i + args.bs * 50], key=lambda d: len(d[0]) + len(d[1])) for i in range(0, len(data), args.bs * 50)]
    batches = [c[i:i + args.bs] for c in chunks for i in range(0, len(c), args.bs)]
    random.shuffle(batches)
    tot, nb = 0.0, 0
    for BB in batches:
        if step >= steps:
            break
        opt.zero_grad(set_to_none=True)
        lsum = 0.0
        for j in range(0, len(BB), args.micro):
            B = BB[j:j + args.micro]
            ids, att, pos, tgt = batch_tensors(B)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                m_hat = judge_scores_train(inner, ids, att, pos) if args.judge else spec_gaps_train(inner, ids, att, pos, tgt)
            loss = loss_fn(m_hat.float(), B) * len(B) / len(BB)
            loss.backward(); lsum += float(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); step += 1
        tot += lsum; nb += 1
        if nb % 100 == 0:
            print(f"  ep {ep} step {step}/{steps} loss {tot / nb:.4f} scale {float(log_s.exp()):.3f} {time.time() - t0:.0f}s", flush=True)
    print(f"epoch {ep} loss {tot / max(nb, 1):.4f} scale {float(log_s.exp()):.3f} {time.time() - t0:.0f}s", flush=True)

out = f"{RUNS}/models/spec_{args.spec}{args.tag}"
os.makedirs(out, exist_ok=True)
if args.lora:
    lm.save_pretrained(out)
else:
    torch.save({k: v for k, v in lm.state_dict().items() if "embed_tokens" not in k and "lm_head" not in k}, f"{out}/full.pt")
jdump({"spec": args.spec, "scale": float(log_s.exp()), "lora": args.lora, "examples": len(data), "epochs": args.epochs,
       "binary": args.binary, "judge": args.judge, "trusted": args.trusted, "nosynth": args.nosynth, "limit": args.limit,
       "train_seconds": time.time() - t0, "wls": args.wls, "seed": args.seed}, f"{out}/scale.json")
print("saved", out, flush=True)
