"""LLM counterfactual teacher signals for (hop, inserted document) pairs.

For every reachable pair (h, d) of a replayed stream the teacher provides
  y   : does inserting d alone into hop h's top-k change the hop answer?    (defeat label)
  r,v : reading d alone, does it answer q_h, and with which value?          (slot relevance / asserted value)
Unreachable pairs are added as negatives (label known exactly by Thm. 1, no LLM call).
"""
import argparse, random
import numpy as np
from common import *
from rag import HOP, clean_ans

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--split", default="train")
ap.add_argument("--neg_ratio", type=float, default=1.0)
ap.add_argument("--max_pairs", type=int, default=60000)
args = ap.parse_args()
rng = random.Random(0)
R = pload(f"{RUNS}/{args.wl}/{args.split}/oracle.pkl")
k = R["k"]
E = torch.tensor(R["E"]).float().cuda()
qemb = {q: torch.tensor(v).float().cuda() for q, v in R["qemb"].items()}
texts = R["texts"]

READ = ("Read the document and answer the question using only this document. If the document does not state the answer, "
        "reply exactly 'unknown'. Reply with only the answer.\n\nDocument: {d}\n\nQuestion: {q}\nAnswer:")


def version_before(cid, b):
    v = None
    for vb, r in R["versions"][cid]:
        if vb < b:
            v = r
    return v


pairs = []          # (cid, b, h, row, reachable)
for lg in R["logs"]:
    b = lg["b"]
    ins = [(row, kind) for row, kind, doc in lg["ins"]]
    ins_rows = [r for r, kd in ins if kd not in ("modify_benign", "modify_value")]
    reach_set = set()
    for cid, info in lg["reach"].items():
        for h, row in info["ins"]:
            if row in ins_rows:
                pairs.append((cid, b, h, row, True)); reach_set.add((cid, h, row))
    # unreachable negatives sampled from the same batch
    if ins_rows:
        cids = list(R["versions"].keys())
        nneg = int(args.neg_ratio * sum(1 for p in pairs if p[1] == b))
        for _ in range(nneg):
            cid = rng.choice(cids); v = version_before(cid, b)
            if not v or not v["trace"]:
                continue
            h = rng.randrange(len(v["trace"])); row = rng.choice(ins_rows)
            if (cid, h, row) not in reach_set:
                pairs.append((cid, b, h, row, False))
rng.shuffle(pairs)
pairs = pairs[: args.max_pairs]
print("pairs", len(pairs), "reachable", sum(p[4] for p in pairs), flush=True)

llm = LLMWrap(mem=0.55)
recs, cf_msgs, rd_msgs, idx_r = [], [], [], []
for cid, b, h, row, reach in pairs:
    hop = version_before(cid, b)["trace"][h]
    rec = {"cid": cid, "b": b, "h": h, "row": row, "reach": reach, "subq": hop["subq"], "ans": hop["ans"],
           "doc": texts[row], "margin": float(qemb[hop["subq"]] @ E[row] - hop["tau"])}
    if reach:
        s_new = float(qemb[hop["subq"]] @ E[row])
        cand = sorted(list(zip(hop["scores"], hop["rows"])) + [(s_new, row)], key=lambda x: -x[0])[:k]
        docs = "\n".join(f"[{i+1}] {texts[r]}" for i, (_, r) in enumerate(cand))
        cf_msgs.append(HOP.format(docs=docs, q=hop["subq"]))
        rd_msgs.append(READ.format(d=texts[row], q=hop["subq"]))
        idx_r.append(len(recs))
    else:
        rec.update({"y": 0, "cf_ans": hop["ans"]})
    recs.append(rec)
cf = llm.chat(cf_msgs, max_tokens=32)
rd = llm.chat(rd_msgs, max_tokens=32)
for i, a, v in zip(idx_r, cf, rd):
    a, v = clean_ans(a), clean_ans(v)
    recs[i]["cf_ans"] = a
    recs[i]["y"] = int(not same_answer(a, recs[i]["ans"]))
    recs[i]["read"] = v
    recs[i]["r"] = int(normalize(v) not in ("unknown", "", "none", "not stated", "not mentioned"))
for r in recs:
    r.setdefault("r", 0); r.setdefault("read", "unknown")
ys = [r["y"] for r in recs if r["reach"]]
print("reachable pairs", len(ys), "defeat rate", np.mean(ys) if ys else 0, "slot-relevant", np.mean([r["r"] for r in recs if r["reach"]]), flush=True)
jdump(recs, f"{RUNS}/{args.wl}/{args.split}/pairs.json")
