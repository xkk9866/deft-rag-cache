"""Interactive cost of one reader call: per-request latency of hop re-reads and full multi-hop recomputations
with the INT4 reader on one GPU (requests issued one at a time, as on a cache miss / certification)."""
import argparse, time
import numpy as np
from common import *
from simulate import Oracle
from rag import RAG, HOP

ap = argparse.ArgumentParser(); ap.add_argument("--wl", default="wiki_cf"); args = ap.parse_args()
O = Oracle(args.wl, "test")
llm = LLMWrap()
hops = [O.trace(c, 0)[h] for c in O.cids[:300] for h in range(len(O.trace(c, 0)))][:200]
lat_hop = []
for hop in hops:
    docs = "\n".join(f"[{i+1}] {O.texts[r]}" for i, r in enumerate(hop["rows"]))
    t = time.time(); llm.chat([HOP.format(docs=docs, q=hop["subq"])], max_tokens=32); lat_hop.append(time.time() - t)
enc = Encoder("cuda:0"); index = DenseIndex(cap=len(O.texts) + 10)
alive = O.alive[0].cpu().numpy(); ids = [i for i in range(len(O.texts)) if alive[i]]
index.add([O.doc_of_row[i] for i in ids], [O.texts[i] for i in ids], O.E[ids].half())
rag = RAG(llm, enc, index, k=O.k)
lat_full = []
for c in O.cids[:60]:
    t = time.time(); rag.run([O.case[c]["q"]]); lat_full.append(time.time() - t)
res = {"hop_reread_ms_p50": 1000 * float(np.median(lat_hop)), "hop_reread_ms_mean": 1000 * float(np.mean(lat_hop)),
       "hop_reread_ms_p95": 1000 * float(np.percentile(lat_hop, 95)), "full_recompute_ms_p50": 1000 * float(np.median(lat_full)),
       "full_recompute_ms_p95": 1000 * float(np.percentile(lat_full, 95))}
print(res, flush=True)
jdump(res, f"{RUNS}/bench/latency_{args.wl}.json")
