"""Which defeated entries get low DEFT scores? (one-step, test stream)"""
import sys, collections
from simulate import *
from deft import load_model
wl = sys.argv[1]
O = Oracle(wl, "test"); m = load_model("svde"); sc = Scorer(O, "svde", model=m)
rows = []
for b in range(1, O.nB + 1):
    held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
    dfd = [c for c, x in O.logs[b - 1]["changed"].items() if x]
    if not dfd:
        continue
    sub = {c: held[c] for c in dfd}
    pairs, prov = reach_pairs(O, sub, b)
    s = sc.score_pairs(pairs)
    best = {}
    for ((c, v, h), r), x in zip(pairs, s):
        if c not in best or x > best[c][0]:
            best[c] = (x, h, r)
    for c in dfd:
        if c in prov:
            continue
        x, h, r = best.get(c, (-9, -1, -1))
        v = held[c]; nv = O.cur_vidx(c, b)
        kinds = sorted({O.ins_kind.get(rr, "?") for (cc, vv, hh), rr in pairs if cc == c})
        rows.append((x, b, c, kinds, O.ans(c, v), O.ans(c, nv), O.trace(c, v)[h]["subq"] if h >= 0 else "", O.texts[r][:120] if r >= 0 else ""))
rows.sort(key=lambda t: t[0])
print("n defeats (non-prov)", len(rows))
kc = collections.Counter(tuple(r[3]) for r in rows[: len(rows) // 5]); print("kinds of lowest 20%:", kc.most_common(8))
for r in rows[:25]:
    print(f"{r[0]:.2f} b{r[1]} {r[3]} | {r[4][:30]!r} -> {r[5][:30]!r} | hop: {r[6][:60]} | doc: {r[7]}")
