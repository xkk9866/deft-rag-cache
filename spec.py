"""Speculator: a smaller reader of the same model family that *measures* the certificate instead of predicting it.

A cached hop answer y = (y_1..y_n, <stop>) was produced greedily by the reader G on evidence K*.  Whether G still
produces y on the current evidence K is decided by the signed margin  m_G(y; K) = min_i [log p_G(y_i | x_K, y_<i) -
max_{v != y_i} log p_G(v | x_K, y_<i)]  (Prop. spec).  Measuring m_G costs a reader pass.  The speculator g is a much
smaller model of the same family (same tokenizer, same chat template, same prompt).  It teacher-forces the *same*
cached tokens on the *same* prompt and reports its own signed margin m_g(y; K); the difference of logits at the
answer positions is the margin, so one forward pass over [x_K; y] -- no decoding, no head, no training -- gives it.
Because g and G read the same prompt with the same vocabulary, m_g is a native, zero-shot predictor of m_G.
Margin distillation (spec_train.py) fine-tunes g so that m_g matches the measured m_G.

Only the answer positions are projected through the LM head, so the full-vocabulary logits are never materialised
for the prompt.
"""
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from common import *
from rag import HOP

SPEC_PATHS = {
    "q06": f"{ROOT}/models/Qwen3-0.6B", "q17": f"{ROOT}/models/Qwen3-1.7B", "q4": f"{ROOT}/models/Qwen3-4B",
    "q4awq": "/home/yaokang/models/Qwen3-4B-AWQ",
    "l1": f"{ROOT}/models/Llama-3.2-1B-Instruct", "l3": f"{ROOT}/models/Llama-3.2-3B-Instruct",
}


def hop_prompt(tok, texts, q, rows):
    """exactly the reader's prompt (rag.HOP inside the chat template, thinking off)"""
    docs = "\n".join(f"[{i + 1}] {texts[r]}" for i, r in enumerate(rows))
    msgs = [{"role": "user", "content": HOP.format(docs=docs, q=q)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class Speculator:
    def __init__(self, name, adapter="", dtype=torch.bfloat16, device="cuda", max_len=3072):
        path = SPEC_PATHS.get(name, name)
        self.name, self.max_len, self.device = name, max_len, device
        self.tok = AutoTokenizer.from_pretrained(path)
        self.lm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa").to(device).eval()
        self.tok_budget = int(os.environ.get("SPEC_TOK_BUDGET", 12000))   # tokens per forward batch (bounds activation memory)
        if adapter:
            ad = adapter if os.path.isabs(adapter) else f"{RUNS}/models/{adapter}"
            if os.path.exists(f"{ad}/adapter_config.json"):         # LoRA adapter (margin distillation)
                from peft import PeftModel
                self.lm = PeftModel.from_pretrained(self.lm, ad).merge_and_unload().eval()
            else:                                                   # full fine-tune state dict
                self.lm.load_state_dict(torch.load(f"{ad}/full.pt", map_location=device), strict=False)
            self.lm = self.lm.to(dtype)
            self.scale = jload(f"{ad}/scale.json").get("scale", 1.0) if os.path.exists(f"{ad}/scale.json") else 1.0
        self.memo, self.pmemo = {}, {}
        self.time, self.calls, self.tokens = 0.0, 0, 0

    def prompt_ids(self, texts, q, rows, ns=""):
        key = (ns, q, tuple(rows))
        if key not in self.pmemo:
            self.pmemo[key] = self.tok(hop_prompt(self.tok, texts, q, rows), add_special_tokens=False)["input_ids"]
        return self.pmemo[key]

    @torch.no_grad()
    def margins(self, texts, items, bs=16, return_gaps=False, ns=""):
        """items: [(q, rows, y_tokens)] -> signed margins m_g(y; rows) (and per-position gaps if requested).
        ns: namespace of the row ids (different oracle replays number their rows differently)."""
        keys = [(ns,) + tuple(it) for it in items]
        need = [k for k in dict.fromkeys(keys) if k not in self.memo]
        if need:
            t = time.time()
            seqs = [(self.prompt_ids(texts, q, rows, ns), list(y)) for _, q, rows, y in need]
            order = sorted(range(len(seqs)), key=lambda i: -(len(seqs[i][0]) + len(seqs[i][1])))
            i = 0
            while i < len(order):
                L = len(seqs[order[i]][0]) + len(seqs[order[i]][1])      # longest first: batch size bounded by the token budget
                n = max(1, min(bs, self.tok_budget // max(1, L)))
                idx = order[i:i + n]; i += n
                res = self._forward([seqs[j] for j in idx])
                for j, (m, g) in zip(idx, res):
                    self.memo[need[j]] = (m, g)
            self.time += time.time() - t; self.calls += len(need)
        out = [self.memo[k] for k in keys]
        return out if return_gaps else [m for m, _ in out]

    def _forward(self, seqs):
        """one teacher-forced pass per batch; LM head only on the answer positions"""
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        L = [min(len(p) + len(y), self.max_len) for p, y in seqs]
        T = max(L)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        pos, tgt = [], []
        for b, (p, y) in enumerate(seqs):
            s = (p + y)
            if len(s) > self.max_len:                               # keep the tail (question + answer) if ever too long
                cut = len(s) - self.max_len; p = p[cut:]; s = s[cut:]
            ids[b, :len(s)] = torch.tensor(s); att[b, :len(s)] = 1
            pos.append([len(p) - 1 + i for i in range(len(y))]); tgt.append(list(y))
        ids, att = ids.to(self.device), att.to(self.device)
        self.tokens += int(att.sum())
        base = self.lm.get_decoder() if hasattr(self.lm, "get_decoder") else self.lm.model
        h = base(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state     # prefill only: no KV cache
        out = []
        for b in range(len(seqs)):
            hp = h[b, pos[b]]                                        # (n, H)
            z = self.lm.lm_head(hp).float()                           # (n, V)
            y = torch.tensor(tgt[b], device=z.device)
            lp = torch.log_softmax(z, -1).gather(1, y[:, None]).squeeze(1)   # log p(y_i | prefix): likelihood of the cached tokens
            zy = z.gather(1, y[:, None]).squeeze(1)
            z.scatter_(1, y[:, None], float("-inf"))
            gaps = (zy - z.max(1).values)
            out.append((float(gaps.min()), {"gaps": gaps.cpu().numpy(), "ll": float(lp.sum()), "ll_min": float(lp.min())}))
        return out

    def stats(self, texts, items, bs=16, ns=""):
        """per item: {'m': min gap, 'll': sum log p(y), 'll_min': min_i log p(y_i), 'gaps': per-position gaps}"""
        res = self.margins(texts, items, bs=bs, return_gaps=True, ns=ns)
        return [dict(m=m, **g) for m, g in res]

    # ------------------------------------------------------------------ the obvious alternatives, same model
    JUDGE = ("Answer the question using the documents below. Each document starts with its year in parentheses; when documents "
             "disagree, trust the most recent one.\n\nDocuments:\n{docs}\n\nQuestion: {q}\nProposed answer: {a}\n"
             "Is the proposed answer correct according to the documents? Reply with yes or no.")

    def judge_prompt_ids(self, texts, q, rows, a):
        docs = "\n".join(f"[{i + 1}] {texts[r]}" for i, r in enumerate(rows))
        msgs = [{"role": "user", "content": self.JUDGE.format(docs=docs, q=q, a=a)}]
        s = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return self.tok(s, add_special_tokens=False)["input_ids"]

    def _yes_no_ids(self):
        if not hasattr(self, "_yn"):
            yes = {self.tok(v, add_special_tokens=False)["input_ids"][0] for v in ("yes", "Yes", " yes", " Yes")}
            no = {self.tok(v, add_special_tokens=False)["input_ids"][0] for v in ("no", "No", " no", " No")}
            self._yn = (sorted(yes), sorted(no))
        return self._yn

    @torch.no_grad()
    def judge(self, texts, items, bs=16, ns=""):
        """zero-shot judge: the same model is asked whether the cached answer is still correct; returns
        log P(yes) - log P(no) at the first generated position (one forward pass, no generation)"""
        keys = [("J", ns) + tuple(it) for it in items]
        need = [k for k in dict.fromkeys(keys) if k not in self.memo]
        if need:
            t = time.time()
            yes, no = self._yes_no_ids()
            seqs = [self.judge_prompt_ids(texts, q, rows, a) for _, _, q, rows, a in need]
            order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
            i = 0
            while i < len(order):
                L = len(seqs[order[i]]); n = max(1, min(bs, self.tok_budget // max(1, L)))
                idx = order[i:i + n]; i += n
                z = self._last_logits([seqs[j] for j in idx])
                lz = torch.log_softmax(z, -1)
                sc = (torch.logsumexp(lz[:, yes], -1) - torch.logsumexp(lz[:, no], -1)).tolist()
                for j, s in zip(idx, sc):
                    self.memo[need[j]] = s
            self.time += time.time() - t; self.calls += len(need)
        return [self.memo[k] for k in keys]

    def _last_logits(self, seqs):
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), pad, dtype=torch.long); att = torch.zeros((len(seqs), T), dtype=torch.long)
        for b, s in enumerate(seqs):
            ids[b, T - len(s):] = torch.tensor(s); att[b, T - len(s):] = 1           # left padding: last position aligned
        ids, att = ids.to(self.device), att.to(self.device)
        self.tokens += int(att.sum())
        base = self.lm.get_decoder() if hasattr(self.lm, "get_decoder") else self.lm.model
        h = base(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state[:, -1]
        return self.lm.lm_head(h).float()

    @torch.no_grad()
    def reanswer(self, texts, items, bs=16, ns="", max_new=24):
        """zero-shot re-answering: the same model answers the hop on the current evidence; returns its answer string
        (greedy).  The comparison with the cached answer is left to the caller."""
        keys = [("R", ns) + tuple(it) for it in items]
        need = [k for k in dict.fromkeys(keys) if k not in self.memo]
        if need:
            t = time.time()
            self.tok.padding_side = "left"
            seqs = [self.prompt_ids(texts, q, rows, ns) for _, _, q, rows in need]
            order = sorted(range(len(seqs)), key=lambda i: -len(seqs[i]))
            pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
            i = 0
            while i < len(order):
                L = len(seqs[order[i]]); n = max(1, min(bs, self.tok_budget // max(1, L + max_new)))
                idx = order[i:i + n]; i += n
                T = max(len(seqs[j]) for j in idx)
                ids = torch.full((len(idx), T), pad, dtype=torch.long); att = torch.zeros((len(idx), T), dtype=torch.long)
                for b, j in enumerate(idx):
                    ids[b, T - len(seqs[j]):] = torch.tensor(seqs[j]); att[b, T - len(seqs[j]):] = 1
                out = self.lm.generate(input_ids=ids.to(self.device), attention_mask=att.to(self.device), max_new_tokens=max_new,
                                       do_sample=False, pad_token_id=pad, eos_token_id=self.tok.eos_token_id)
                self.tokens += int(att.sum())
                for b, j in enumerate(idx):
                    txt = self.tok.decode(out[b, T:], skip_special_tokens=True)
                    self.memo[need[j]] = txt.strip().split("\n")[0].strip().strip('"').rstrip(".")[:120]
            self.time += time.time() - t; self.calls += len(need)
        return [self.memo[k] for k in keys]


def spec_gaps_train(lm, ids, att, pos, tgt):
    """differentiable version for margin distillation: returns per-example margins (min gap over answer positions)"""
    base = lm.get_decoder() if hasattr(lm, "get_decoder") else lm.model
    h = base(input_ids=ids, attention_mask=att, use_cache=False).last_hidden_state
    ms = []
    for b in range(ids.shape[0]):
        z = lm.lm_head(h[b, pos[b]]).float()
        y = tgt[b]
        zy = z.gather(1, y[:, None]).squeeze(1)
        zo = z.scatter(1, y[:, None], float("-inf")).max(1).values
        ms.append((zy - zo).min())
    return torch.stack(ms)
