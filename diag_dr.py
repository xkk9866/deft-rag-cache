"""diagnose Delta-Reader on a workload: AUC of entry scores and the lowest-scored defeated hops."""
import sys, numpy as np
from common import *
from simulate import *
from deltar import load_dr, score_examples
from sklearn.metrics import roc_auc_score
wl = sys.argv[1]; var = sys.argv[2] if len(sys.argv) > 2 else "full"
O = Oracle(wl, "test")
m = load_dr(var)
sc = SetScorer(O, m)
ys, ss, info = [], [], []
for b in range(1, O.nB + 1):
    held = {c: O.cur_vidx(c, b - 1) for c in O.cids}
    s = sc.entry_scores(held, b)
    for c in O.logs[b - 1]["reach"]:
        ys.append(int(bool(O.logs[b - 1]["changed"].get(c)))); ss.append(s[c]); info.append((b, c, held[c]))
ys, ss = np.array(ys), np.array(ss)
print("entries", len(ys), "pos", ys.sum(), "AUC", roc_auc_score(ys, ss))
print("score quantiles pos", np.quantile(ss[ys == 1], [0, .01, .02, .05, .1, .5]), "neg", np.quantile(ss[ys == 0], [.1, .5, .9]))
print("params beta gamma k1 k2 c", m.log_beta.exp().item(), m.log_gamma.exp().item(), m.k1.item(), m.k2.item(), m.c.item())
low = np.argsort(np.where(ys == 1, ss, np.inf))[:12]
for i in low:
    b, c, v = info[i]
    vn = O.cur_vidx(c, b)
    for h, (ho, hn) in enumerate(zip(O.trace(c, v), O.trace(c, vn))):
        if ho["rows"] != hn["rows"] or ho["ans"] != hn["ans"]:
            z = sc.memo.get(((c, v, h), b))
            print(f"--- score {ss[i]:.2f} b{b} hop{h} z={z} Q: {ho['subq']} | old: {ho['ans']} | new: {hn['ans']} | final {O.ans(c, v)} -> {O.ans(c, vn)}")
            new_rows = [x for x in hn["rows"] if x not in ho["rows"]]; gone = [x for x in ho["rows"] if x not in hn["rows"]]
            for x in new_rows: print("   +", O.ins_kind.get(x, "?"), O.texts[x][:150])
            for x in gone: print("   -", O.texts[x][:150])
            break
