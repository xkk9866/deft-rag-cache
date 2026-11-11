import collections, random
from common import *
random.seed(0)
for wl in ["wiki_cf", "hotpot"]:
    R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
    kind = {row: k for lg in R["logs"] for row, k, _ in lg["ins"]}
    P = [p for p in jload(f"{RUNS}/{wl}/train/pairs.json") if p["reach"] and p["y"] and kind.get(p["row"]) in ("random", "sibling", "restate", "update_dup")]
    print("=====", wl, len(P))
    for p in random.sample(P, 14):
        print(f"  [{kind[p['row']]}] A: {p['ans'][:50]!r} -> CF: {p['cf_ans'][:60]!r}")
