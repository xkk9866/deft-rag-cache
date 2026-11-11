"""Diagnostic: how much does the reader's trusted margin say about whether an evidence edit defeats a hop?
Hop-level edits of the cal/test replays (first changed hop per recomputed entry, as in deltar.edit_examples).
Reports AUCs of margin-only, DEFT-X-only and their combination (logistic fit on cal, evaluated on test) and the
recall-matched saving (share of non-defeating edits passed at 90/95% recall of defeats)."""
import argparse, sys
import numpy as np
from common import *
from mcert import MarginBook
from deltar import edit_examples, load_dr, score_examples

ap = argparse.ArgumentParser()
ap.add_argument("--wls", default="wiki_t,wiki_cf,hotpot,wiki_rev")
ap.add_argument("--score", action="store_true")
args = ap.parse_args()


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def save_at(s, y, rec):
    s, y = np.asarray(s, float), np.asarray(y, int)
    thr = np.sort(s[y == 1])[int(np.floor((1 - rec) * (y == 1).sum()))]
    return float(np.mean(s[y == 0] < thr))


def feats(wl, split, model=None):
    exs, T = edit_examples(wl, split)
    MB = MarginBook(wl, split)
    rows = []
    for e in exs:
        K = [x for _, x in e["K"]]; K2 = [x for _, x in e["K2"]]
        if not MB.has(e["subq"], K) or not MB.has(e["subq"], K2):
            continue
        m2, cen = MB.post(e["subq"], K, K2)
        rows.append({"y": e["y"], "mu": MB.mu(e["subq"], K), "m2": m2, "cen": cen, "ex": e})
    if model is not None:
        z = score_examples(model, [r["ex"] for r in rows], {}, lambda d: T[d[1]], bs=16)
        for r, x in zip(rows, z):
            r["z"] = float(x)
    return rows


model = load_dr("setcross1", "_m3") if args.score else None
for wl in args.wls.split(","):
    te = feats(wl, "test", model)
    y = np.array([r["y"] for r in te]); mu_ = np.array([r["mu"] for r in te]); m2 = np.array([r["m2"] for r in te])
    tok = (m2 < 0).astype(int)
    print(f"== {wl} test edits {len(te)} defeated {y.mean():.3f} token-flip {tok.mean():.3f} "
          f"agree(equiv-label, token-flip) {np.mean(tok == y):.3f} censored {np.mean([r['cen'] for r in te]):.3f}")
    print(f"   AUC margin-only {auc(-mu_, y):.3f}   (oracle post-margin {auc(-m2, y):.3f})")
    for q in (0.5, 1, 2, 4, 8):
        sel = mu_ < q
        print(f"   mu<{q}: share {sel.mean():.3f}  defeat rate {y[sel].mean() if sel.any() else 0:.3f} | mu>={q}: {y[~sel].mean() if (~sel).any() else 0:.3f}")
    if model is not None:
        ca = feats(wl, "cal", model)
        z = np.array([r["z"] for r in te])
        Xc = np.array([[r["z"], np.log1p(r["mu"])] for r in ca]); yc = np.array([r["y"] for r in ca])
        Xt = np.stack([z, np.log1p(mu_)], 1)
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=10.0).fit(Xc, yc)
        comb = lr.decision_function(Xt)
        print(f"   AUC DEFT-X {auc(z, y):.3f}  combined {auc(comb, y):.3f}  coef {lr.coef_.round(3).tolist()}")
        for rec in (0.9, 0.95):
            print(f"   save@recall{rec}: margin {save_at(-mu_, y, rec):.3f}  DEFT-X {save_at(z, y, rec):.3f}  combined {save_at(comb, y, rec):.3f}")
    sys.stdout.flush()
