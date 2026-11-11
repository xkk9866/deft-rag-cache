"""Speculative certification: the cached trace is a free draft, and certifying it is draft verification.

Greedy decoding returns the token sequence y iff every y_i is the argmax of p(. | prompt, y_<i) and the stop token
follows.  All these conditions are read off ONE teacher-forced forward pass over (prompt(q_j, K'_j), y), exactly the
verification step of speculative decoding, with the cached hop answer as the draft.  So a certification re-read needs
no autoregressive decoding: accept => the hop is provably unchanged (token-exact, strictly stronger than any answer
equivalence); reject at position i => the reader's own argmax at i starts the new answer and only the suffix is decoded.

For every hop that hop-local certification re-reads on the test stream (subq unchanged, evidence changed):
  draft  = greedy tokens on the trusted evidence K   (what a deployed cache stores)
  truth  = greedy tokens on the current evidence K'  (what recomputation would produce)
  verify = one teacher-forced pass on (K', draft) with prompt log-probs
Reports verdict agreement (accept <=> truth == draft), accepted-prefix length on rejects, batched GPU time and
single-request latency of re-generation vs. verification.  Output: runs/<wl>/speccert/results.json
"""
import argparse, time, random
import numpy as np
from common import *
from simulate import Oracle, certify_cost
from rag import HOP, clean_ans

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--max_hops", type=int, default=3000)
ap.add_argument("--lat_n", type=int, default=100)
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/speccert"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")

# ---- the hops certification re-reads: subq unchanged and evidence changed, from the trusted to the current version
items = []
for b in range(1, O.nB + 1):
    for c in O.logs[b - 1]["reach"]:
        vo, vn = O.cur_vidx(c, b - 1), O.cur_vidx(c, b)
        to, tn = O.trace(c, vo), O.trace(c, vn)
        for j, (ho, hn) in enumerate(zip(to, tn)):
            if ho["subq"] != hn["subq"]:
                break
            if ho["rows"] != hn["rows"]:
                items.append((ho["subq"], tuple(ho["rows"]), tuple(hn["rows"])))
items = list(dict.fromkeys(items))
random.Random(0).shuffle(items)
items = items[: args.max_hops]
print("hops to certify", len(items), flush=True)

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
mem = float(os.environ.get("LLM_MEM", 0.6))
llm = LLM(model=LLM_PATH, quantization="awq_marlin", gpu_memory_utilization=mem, max_model_len=int(os.environ.get("LLM_MAXLEN", 4096)),
          seed=0, enable_prefix_caching=False, max_num_seqs=int(os.environ.get("LLM_SEQS", 32)), enforce_eager=bool(os.environ.get("LLM_EAGER")),
          **({"max_num_batched_tokens": int(os.environ["LLM_BTOK"])} if os.environ.get("LLM_BTOK") else {}))
tok = llm.get_tokenizer()
STOP_IDS = {i for i in [tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")] if isinstance(i, int) and i >= 0}


def prompt_ids(q, rows):
    docs = "\n".join(f"[{i + 1}] {O.texts[r]}" for i, r in enumerate(rows))
    s = tok.apply_chat_template([{"role": "user", "content": HOP.format(docs=docs, q=q)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


GEN = SamplingParams(temperature=0, max_tokens=32)
VER = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=1)


def gen_tokens(outs):
    res = []
    for o in outs:
        t = list(o.outputs[0].token_ids)
        if o.outputs[0].finish_reason == "stop" and (not t or t[-1] not in STOP_IDS):
            t.append(next(iter(STOP_IDS)))
        res.append(t)
    return res


P_old = [prompt_ids(q, k) for q, k, _ in items]
P_new = [prompt_ids(q, k2) for q, _, k2 in items]
t0 = time.time(); drafts = gen_tokens(llm.generate([TokensPrompt(prompt_token_ids=p) for p in P_old], GEN, use_tqdm=False)); t_draft = time.time() - t0
t0 = time.time(); truth = gen_tokens(llm.generate([TokensPrompt(prompt_token_ids=p) for p in P_new], GEN, use_tqdm=False)); t_gen = time.time() - t0
t0 = time.time(); vouts = llm.generate([TokensPrompt(prompt_token_ids=p + d) for p, d in zip(P_new, drafts)], VER, use_tqdm=False); t_ver = time.time() - t0


def verdict(o, p, d):
    """accept iff every draft token is the reader's argmax; on reject return the accepted prefix length"""
    pl = o.prompt_logprobs
    for i, tkn in enumerate(d):
        lp = pl[len(p) + i]
        if lp is None or tkn not in lp or lp[tkn].rank != 1:
            return False, i
    return True, len(d)


agree = fa = fr = 0; acc_frac, n_rej, dec_saved, dec_full = [], 0, 0, 0
lax_accept = 0
for (q, _, _), p, d, g, o in zip(items, P_new, drafts, truth, vouts):
    ok, i = verdict(o, p, d)
    same = (g == d)
    agree += int(ok == same); fa += int(ok and not same); fr += int((not ok) and same)
    dec_full += len(g)
    if ok:
        dec_saved += len(g)
    else:
        n_rej += 1; acc_frac.append(i / max(1, len(d)))
        dec_saved += min(i, len(g))
    da, ga = clean_ans(tok.decode(d, skip_special_tokens=True)), clean_ans(tok.decode(g, skip_special_tokens=True))
    lax_accept += int(equiv(da, ga) and not same)
n = len(items)
res = {"wl": args.wl, "n_hops": n, "agreement": agree / n, "false_accept": fa, "false_reject": fr,
       "changed_rate": float(np.mean([g != d for g, d in zip(truth, drafts)])),
       "reject_accepted_prefix": float(np.mean(acc_frac)) if acc_frac else 0.0,
       "decode_tokens_saved": dec_saved / max(1, dec_full),
       "lax_equiv_but_token_diff": lax_accept / n,
       "batch_gen_ms_per_hop": 1000 * t_gen / n, "batch_ver_ms_per_hop": 1000 * t_ver / n,
       "mean_answer_tokens": float(np.mean([len(g) for g in truth])),
       "mean_prompt_tokens": float(np.mean([len(p) for p in P_new]))}
print(res, flush=True)

# ---- single-request latency (interactive certification): one hop at a time
idx = list(range(min(args.lat_n, n)))
lg, lv = [], []
for i in idx:
    t0 = time.time(); llm.generate([TokensPrompt(prompt_token_ids=P_new[i])], GEN, use_tqdm=False); lg.append(1000 * (time.time() - t0))
    t0 = time.time(); llm.generate([TokensPrompt(prompt_token_ids=P_new[i] + drafts[i])], VER, use_tqdm=False); lv.append(1000 * (time.time() - t0))
res.update({"lat_gen_p50": float(np.percentile(lg, 50)), "lat_gen_p95": float(np.percentile(lg, 95)),
            "lat_ver_p50": float(np.percentile(lv, 50)), "lat_ver_p95": float(np.percentile(lv, 95))})

# ---- whole-entry certification: all re-read hops of a suspect entry verified concurrently (their inputs are all known
# from the cache and the current retrieval state), versus the sequential re-read of hop-local certification
ent = {}
for b in range(1, O.nB + 1):
    for c in O.logs[b - 1]["reach"]:
        vo, vn = O.cur_vidx(c, b - 1), O.cur_vidx(c, b)
        to, tn = O.trace(c, vo), O.trace(c, vn)
        hs = []
        for j, (ho, hn) in enumerate(zip(to, tn)):
            if ho["subq"] != hn["subq"]:
                break
            if ho["rows"] != hn["rows"]:
                hs.append((ho["subq"], tuple(ho["rows"]), tuple(hn["rows"])))
        if len(hs) >= 2:
            ent[(c, b)] = hs
multi = list(ent.values())[: args.lat_n]
ls, lp = [], []
for hs in multi:
    pn = [prompt_ids(q, k2) for q, _, k2 in hs]
    dr = gen_tokens(llm.generate([TokensPrompt(prompt_token_ids=prompt_ids(q, k)) for q, k, _ in hs], GEN, use_tqdm=False))
    t0 = time.time()
    for p in pn:
        llm.generate([TokensPrompt(prompt_token_ids=p)], GEN, use_tqdm=False)
    ls.append(1000 * (time.time() - t0))
    t0 = time.time(); llm.generate([TokensPrompt(prompt_token_ids=p + d) for p, d in zip(pn, dr)], VER, use_tqdm=False); lp.append(1000 * (time.time() - t0))
res.update({"n_multi_hop_entries": len(multi), "multi_mean_hops": float(np.mean([len(h) for h in multi])) if multi else 0.0,
            "entry_seq_regen_p50": float(np.percentile(ls, 50)) if ls else 0.0, "entry_par_verify_p50": float(np.percentile(lp, 50)) if lp else 0.0,
            "entry_seq_regen_p95": float(np.percentile(ls, 95)) if ls else 0.0, "entry_par_verify_p95": float(np.percentile(lp, 95)) if lp else 0.0})
print(res, flush=True)
jdump(res, f"{OUT}/results.json")
print("done", flush=True)
