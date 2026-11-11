"""Certification as margin measurement, implemented without whole-prompt log-probabilities.

Greedy decoding on x returns the cached tokens y = (y_1..y_n, <stop>) iff y_i = argmax p(. | x, y_<i) for all i, and
the signed margin of y is the smallest top-1 minus runner-up gap along y.  The n+1 conditional distributions are read
by n+1 one-token requests  x, x+y_1, ..., x+y_<=n  that share the prompt x through the engine's prefix cache:
phase 1 prefills x once (and already checks y_1), phase 2 runs the n remaining short suffixes concurrently.  Only
answer positions are scored (top-2 log-probabilities), never the prompt positions.

For every hop that certification re-reads on the test stream (sub-question unchanged, evidence changed):
  draft = the cached greedy tokens on the trusted evidence (stored with the margins), truth = greedy tokens on K'.
Reports verdict agreement with re-generation, the measured signed margin, batched GPU time per hop and single-request
latency for (i) re-generation, (ii) two-phase verification.  Output: runs/<wl>/verify/results.json
"""
import argparse, time, random, collections
import numpy as np
from common import *
from simulate import Oracle
from mcert import MarginBook, signed_margin
from rag import HOP

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--max_hops", type=int, default=2000)
ap.add_argument("--lat_n", type=int, default=100)
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/verify"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")
MB = MarginBook(args.wl, "test")

items = []
for b in range(1, O.nB + 1):
    for c in O.logs[b - 1]["reach"]:
        to, tn = O.trace(c, O.cur_vidx(c, b - 1)), O.trace(c, O.cur_vidx(c, b))
        for ho, hn in zip(to, tn):
            if ho["subq"] != hn["subq"]:
                break
            if ho["rows"] != hn["rows"] and MB.has(ho["subq"], ho["rows"]) and MB.has(hn["subq"], hn["rows"]):
                items.append((ho["subq"], tuple(ho["rows"]), tuple(hn["rows"])))
items = list(dict.fromkeys(items))
random.Random(0).shuffle(items)
items = items[: args.max_hops]
print("hops", len(items), flush=True)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
llm = LLM(model=LLM_PATH, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("LLM_MEM", 0.6)),
          max_model_len=4096, seed=0, enable_prefix_caching=True, max_num_seqs=int(os.environ.get("LLM_SEQS", 128)))
tok = llm.get_tokenizer()


def prompt_ids(q, rows):
    s = tok.apply_chat_template([{"role": "user", "content": HOP.format(docs="\n".join(f"[{i + 1}] {O.texts[r]}" for i, r in enumerate(rows)), q=q)}],
                                tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


GEN = SamplingParams(temperature=0, max_tokens=32)
ONE = SamplingParams(temperature=0, max_tokens=1, logprobs=2)


def verify(prompts, drafts):
    """two-phase verification; returns (accept, first rejected position, signed margin) per hop"""
    out1 = llm.generate([TokensPrompt(prompt_token_ids=p) for p in prompts], ONE, use_tqdm=False)
    reqs, owner = [], []
    for k, (p, d) in enumerate(zip(prompts, drafts)):
        for i in range(1, len(d)):
            reqs.append(TokensPrompt(prompt_token_ids=p + d[:i])); owner.append((k, i))
    out2 = llm.generate(reqs, ONE, use_tqdm=False) if reqs else []
    pos = collections.defaultdict(dict)
    for k, o in enumerate(out1):
        pos[k][0] = o.outputs[0].logprobs[0]
    for (k, i), o in zip(owner, out2):
        pos[k][i] = o.outputs[0].logprobs[0]
    res = []
    for k, d in enumerate(drafts):
        m, rej = 1e9, None
        for i, t in enumerate(d):
            lp = pos[k][i]
            best = max(lp.items(), key=lambda kv: kv[1].logprob)
            if t in lp:
                other = max([v.logprob for tid, v in lp.items() if tid != t] or [-1e4])
                g = lp[t].logprob - other
            else:
                g = -abs(best[1].logprob) - 1.0                # cached token below the runner-up: rejected
            m = min(m, g)
            if g <= 0 and rej is None:
                rej = i
        res.append((rej is None, rej, m))
    return res


P_new = [prompt_ids(q, k2) for q, _, k2 in items]
drafts = [MB.tokens(q, k) for q, k, _ in items]
truth_tok = [MB.tokens(q, k2) for q, _, k2 in items]
llm.reset_prefix_cache()
t0 = time.time(); gen = llm.generate([TokensPrompt(prompt_token_ids=p) for p in P_new], GEN, use_tqdm=False); t_gen = time.time() - t0
llm.reset_prefix_cache()
t0 = time.time(); ver = verify(P_new, drafts); t_ver = time.time() - t0
same = [list(g.outputs[0].token_ids) == d for g, d in zip(gen, drafts)]
acc = [v[0] for v in ver]
ref = [signed_margin(d, MB.S[(q, k2)])[0] for (q, _, k2), d in zip(items, drafts)]
n = len(items)
res = {"wl": args.wl, "n_hops": n, "agreement": float(np.mean([a == s for a, s in zip(acc, same)])),
       "false_accept": int(sum(a and not s for a, s in zip(acc, same))), "false_reject": int(sum(s and not a for a, s in zip(acc, same))),
       "changed_rate": float(1 - np.mean(same)), "sign_agree_stored": float(np.mean([(v[2] > 0) == (r >= 0) for v, r in zip(ver, ref)])),
       "batch_gen_ms_per_hop": 1000 * t_gen / n, "batch_ver_ms_per_hop": 1000 * t_ver / n,
       "mean_answer_tokens": float(np.mean([len(d) for d in drafts])), "mean_prompt_tokens": float(np.mean([len(p) for p in P_new]))}
print(res, flush=True)
lg, lv = [], []
for i in range(min(args.lat_n, n)):
    llm.reset_prefix_cache()
    t0 = time.time(); llm.generate([TokensPrompt(prompt_token_ids=P_new[i])], GEN, use_tqdm=False); lg.append(1000 * (time.time() - t0))
    llm.reset_prefix_cache()
    t0 = time.time(); verify([P_new[i]], [drafts[i]]); lv.append(1000 * (time.time() - t0))
res.update({"lat_gen_p50": float(np.percentile(lg, 50)), "lat_gen_p95": float(np.percentile(lg, 95)),
            "lat_ver_p50": float(np.percentile(lv, 50)), "lat_ver_p95": float(np.percentile(lv, 95))})
print(res, flush=True)
jdump(res, f"{OUT}/results.json")
