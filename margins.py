"""Reader margins: a cached hop answer is a greedy fixed point of the reader, and its decision margin
    mu(y; x) = min_i [ log p(y_i | x, y_<i) - max_{v != y_i} log p(v | x, y_<i) ]      (all answer positions + stop)
is a certificate of how far the evidence can move before greedy decoding leaves y.  It is free: the reader
computes these log-probabilities when it generates the answer and again when a certification verifies it.

For every distinct hop state (sub-question, evidence rows) of the oracle versions of a split -- plus, on the
training split, the single-document counterfactual states of label.py and reader-labelled synthetic edits --
run the reader once (greedy, top-20 log-probs per generated position, stop token included) and store
tokens, top-20 ids and log-probs.  Output: runs/<wl>/<split>/margins.pkl
    {"states": {(subq, rows): {"tok": [...], "ids": int32[L,20], "lp": float16[L,20], "ans": str}},
     "synth": [{"subq", "K": rows, "K2": rows}] }
"""
import argparse, random
import numpy as np
from common import *
from rag import HOP, clean_ans

ap = argparse.ArgumentParser()
ap.add_argument("--jobs", required=True, help="comma list of wl:split")
ap.add_argument("--synth", type=int, default=3, help="synthetic edits per training hop state")
ap.add_argument("--topn", type=int, default=20)
args = ap.parse_args()

llm = LLMWrap(mem=float(os.environ.get("LLM_MEM", 0.8)))
tok = llm.tok
STOP = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")}
        if isinstance(i, int) and i >= 0]
SP = llm.SP(temperature=0, max_tokens=32, logprobs=args.topn)


def prompt(texts, q, rows):
    docs = "\n".join(f"[{i + 1}] {texts[r]}" for i, r in enumerate(rows))
    msgs = [{"role": "user", "content": HOP.format(docs=docs, q=q)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def run_states(texts, keys):
    out = {}
    for i in range(0, len(keys), 2048):
        ch = keys[i:i + 2048]
        outs = llm.llm.generate([prompt(texts, q, r) for q, r in ch], SP, use_tqdm=False)
        for k, o in zip(ch, outs):
            c = o.outputs[0]
            ids = np.zeros((len(c.token_ids), args.topn), dtype=np.int32) - 1
            lp = np.full((len(c.token_ids), args.topn), -1e4, dtype=np.float32)
            for t, d in enumerate(c.logprobs or []):
                items = sorted(d.items(), key=lambda kv: kv[1].rank)[: args.topn]
                for j, (tid, L) in enumerate(items):
                    ids[t, j] = tid; lp[t, j] = L.logprob
            out[k] = {"tok": list(c.token_ids), "ids": ids, "lp": lp.astype(np.float16), "fin": c.finish_reason,
                      "ans": clean_ans(c.text)}
        print(f"  {min(i + 2048, len(keys))}/{len(keys)}", flush=True)
    return out


def synth_edits(R, states, n, rng):
    """reader-labelled counterfactual edits around training hop states: an eviction (one document of K leaves, the
    next corpus document enters), a forced insertion of the most similar document inserted by the training stream,
    and both together.  Inputs of the differential reader stay (sub-question, answer, evidence set)."""
    E = torch.tensor(R["E"]).float().cuda()
    ins = torch.tensor(sorted({row for lg in R["logs"] for row, _, _ in lg["ins"]}), device="cuda")
    out = []
    keys = sorted(states)
    for i in range(0, len(keys), 512):
        ch = keys[i:i + 512]
        Q = torch.stack([torch.tensor(R["qemb"][q]).float().cuda() for q, _ in ch])
        sc, top = torch.topk(Q @ E.T, R["k"] + 20, dim=1)
        si, ti = torch.topk(Q @ E[ins].T, min(8, len(ins)), dim=1)
        for (q, rows), qv, tt, ii in zip(ch, Q, top.tolist(), ti.tolist()):
            K = list(rows)
            rank = lambda xs: tuple(sorted(xs, key=lambda x: -float(qv @ E[x])))      # the reader sees documents in rank order
            cand = [t for t in tt if t not in rows]
            upd = [int(ins[j]) for j in ii if int(ins[j]) not in rows]
            eds = []
            if cand:
                j = rng.randrange(len(K)); eds.append([x for x in K if x != K[j]] + [cand[0]])
            if upd:
                eds.append(K[:-1] + [upd[0]] if upd[0] not in K else K)
                if cand and len(upd) > 1:
                    j = rng.randrange(len(K) - 1); K2 = [x for x in K[:-1] if x != K[j]] + [upd[1], cand[0]]
                    eds.append(K2)
            for K2 in eds[:n]:
                K2 = rank(K2)
                if K2 != tuple(rows):
                    out.append({"subq": q, "K": tuple(rows), "K2": K2})
    return out


for job in args.jobs.split(","):
    wl, split = job.split(":")
    dst = f"{RUNS}/{wl}/{split}/margins.pkl"
    if os.path.exists(dst):
        print("exists", dst, flush=True); continue
    R = pload(f"{RUNS}/{wl}/{split}/oracle.pkl")
    texts = R["texts"]
    states = {}
    for cid, vs in R["versions"].items():
        for _, r in vs:
            for h in r["trace"]:
                states[(h["subq"], tuple(h["rows"]))] = h["ans"]
    keys = list(states)
    synth = []
    if split == "train":
        p = f"{RUNS}/{wl}/train/pairs.json"
        if os.path.exists(p):                                     # single-document counterfactual states (label.py)
            from deltar import version_before
            E = torch.tensor(R["E"]).float()
            for r in jload(p):
                if not r["reach"]:
                    continue
                hop = version_before(R, r["cid"], r["b"])["trace"][r["h"]]
                s_new = float(torch.tensor(R["qemb"][hop["subq"]]).float() @ E[r["row"]])
                cand = sorted(list(zip(hop["scores"], hop["rows"])) + [(s_new, r["row"])], key=lambda x: -x[0])[:R["k"]]
                keys.append((hop["subq"], tuple(x for _, x in cand)))
        if args.synth:
            synth = synth_edits(R, states, args.synth, random.Random(0))
            keys += [(s["subq"], s["K2"]) for s in synth]
    keys = list(dict.fromkeys(keys))
    print(wl, split, "states", len(states), "to read", len(keys), "synthetic edits", len(synth), flush=True)
    res = run_states(texts, keys)
    agree = np.mean([equiv(res[k]["ans"], a) for k, a in states.items()])
    exact = np.mean([res[k]["ans"] == a for k, a in states.items()])
    ex = res[keys[0]]
    print(wl, split, "re-read agrees with the oracle answer: equiv", round(float(agree), 4), "exact", round(float(exact), 4),
          "| sample", ex["tok"], ex["fin"], ex["lp"][:, :2].tolist(), "stop ids", STOP, flush=True)
    pdump({"states": res, "synth": synth, "stop": STOP}, dst)
    print("saved", dst, flush=True)
print("MARGINS DONE", flush=True)
