import collections, sys
from common import *
for wl in ["wiki_t", "wiki_cf", "hotpot"]:
    R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
    kind = {row: k for lg in R["logs"] for row, k, _ in lg["ins"]}
    P = jload(f"{RUNS}/{wl}/train/pairs.json")
    c = collections.defaultdict(lambda: [0, 0, 0])
    for p in P:
        if not p["reach"]:
            continue
        k = kind.get(p["row"], "?"); c[k][0] += 1; c[k][1] += p["y"]; c[k][2] += p["r"]
    print(wl, {k: (v[0], round(v[1] / v[0], 2), round(v[2] / v[0], 2)) for k, v in sorted(c.items())})
    ex = [p for p in P if p["reach"] and kind.get(p["row"]) == "historical"][:3] + [p for p in P if p["reach"] and kind.get(p["row"]) == "update"][:2]
    for p in ex:
        print("  ", kind.get(p["row"]), "| Q:", p["subq"], "| A:", p["ans"], "| CF:", p["cf_ans"], "| read:", p["read"], "| y", p["y"], "|", p["doc"][:110])
