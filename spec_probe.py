"""Feasibility probe: does the zero-shot margin of a same-family sibling predict the reader's defeat on real hop edits?
For every hop edit K* -> K' of a split (oracle replay), with the reader's cached tokens y:
   m_g(y; K')  (sibling, current evidence),  m_g(y; K*)  (sibling, trusted evidence),  mu*_G  (reader, trusted)
and the reader's own signed post-edit margin m_G(y; K') from margins.pkl.  Reports AUC of several predictors and the
rank correlation between m_g and m_G, plus speed.  Optionally scores the trained MTR on the same edits.
"""
import argparse, time, math
import numpy as np
from common import *
from mcert import MarginBook
from deltar import edit_examples
from spec import Speculator

ap = argparse.ArgumentParser()
ap.add_argument("--wls", default="wiki_t")
ap.add_argument("--split", default="test")
ap.add_argument("--spec", default="q06")
ap.add_argument("--adapter", default="")
ap.add_argument("--mtr", default="", help="variant:tag of a trained MTR to score the same edits")
ap.add_argument("--bs", type=int, default=16)
ap.add_argument("--out", default="")
args = ap.parse_args()


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


sp = Speculator(args.spec, args.adapter)
R = {}
for wl in args.wls.split(","):
    exs, texts = edit_examples(wl, args.split)
    MB = MarginBook(wl, args.split)
    rows = []
    for e in exs:
        K, K2 = tuple(x for _, x in e["K"]), tuple(x for _, x in e["K2"])
        y = MB.tokens(e["subq"], K)
        if y is None:
            continue
        m2, cen = MB.post(e["subq"], K, K2) if MB.has(e["subq"], K2) else (None, None)
        rows.append((e["subq"], K, K2, tuple(y), e["y"], MB.mu(e["subq"], K), m2, e["ans"]))
    print(wl, args.split, "hop edits", len(rows), "defeating", sum(r[4] for r in rows), flush=True)
    t = time.time()
    S2 = sp.stats(texts, [(q, K2, y) for q, K, K2, y, *_ in rows], bs=args.bs)
    t_cur = time.time() - t
    S1 = sp.stats(texts, [(q, K, y) for q, K, K2, y, *_ in rows], bs=args.bs)
    lab = np.array([r[4] for r in rows]); mu = np.array([r[5] for r in rows])
    mK2, mK = np.array([s["m"] for s in S2]), np.array([s["m"] for s in S1])
    lK2, lK = np.array([s["ll"] for s in S2]), np.array([s["ll"] for s in S1])
    nK2, nK = np.array([s["ll_min"] for s in S2]), np.array([s["ll_min"] for s in S1])
    res = {"n": len(rows), "pos": int(lab.sum()),
           "auc_spec_cur": auc(-mK2, lab), "auc_spec_did": auc(-(np.log1p(np.maximum(mu, 0)) + mK2 - mK), lab),
           "auc_spec_shift": auc(-(mK2 - mK), lab), "auc_mu": auc(-mu, lab),
           "auc_spec_cur_plus_mu": auc(-(mK2 + np.log1p(np.maximum(mu, 0))), lab),
           "auc_ll_cur": auc(-lK2, lab), "auc_ll_shift": auc(-(lK2 - lK), lab), "auc_llmin_shift": auc(-(nK2 - nK), lab),
           "auc_ll_shift_plus_mu": auc(-(lK2 - lK + np.log1p(np.maximum(mu, 0))), lab),
           "ms_per_hop": 1000 * t_cur / len(rows), "tokens_per_hop": sp.tokens / max(1, sp.calls),
           "sign_agree_trusted": float(np.mean(mK > 0))}
    have = [i for i, r in enumerate(rows) if r[6] is not None]
    if have:
        mG = np.array([rows[i][6] for i in have])
        res["spearman_mg_mG"] = spearman(mK2[have], mG)
        res["sign_agree_mg_mG"] = float(np.mean((mK2[have] < 0) == (mG < 0)))
    if args.mtr:
        from deltar import load_dr, score_examples
        v, tag = (args.mtr.split(":") + [""])[:2]
        M = load_dr(v, tag)
        exs2 = [{"subq": q, "ans": a, "K": [(0, x) for x in K], "K2": [(0, x) for x in K2], "mu": m} for q, K, K2, y, _, m, _, a in rows]
        t = time.time(); z = score_examples(M, exs2, {}, lambda d: texts[d[1]], bs=32); t_m = time.time() - t
        res["auc_mtr"] = auc(z, lab); res["ms_per_hop_mtr"] = 1000 * t_m / len(rows)
        del M; torch.cuda.empty_cache()
    print(wl, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
    R[wl] = res
if args.out:
    jdump(R, args.out)
print("PROBE DONE", flush=True)
