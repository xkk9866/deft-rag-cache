"""Margin statistics per workload: distribution of trusted margins, the re-read disagreement with the recorded oracle
answers (the batched-decoding nondeterminism floor) as a function of the margin, and how well the trusted margin
alone predicts defeat.  Output: runs/margin_stats.json"""
import numpy as np
from common import *
from mcert import MarginBook, mu
from deltar import edit_examples


def auc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


out = {}
for wl in ["wiki_t", "wiki_cf", "hotpot", "wiki_rev", "wiki_t_q4", "wiki_t_ll"]:
    res = {}
    mus, dis, dmu = [], 0, []
    n = 0
    for split in ("cal", "test"):
        R = pload(f"{RUNS}/{wl}/{split}/oracle.pkl")
        MB = MarginBook(wl, split)
        if not MB.S:
            continue
        seen = set()
        for cid, vs in R["versions"].items():
            for _, r in vs:
                for h in r["trace"]:
                    k = (h["subq"], tuple(h["rows"]))
                    if k in seen or k not in MB.S:
                        continue
                    seen.add(k); n += 1
                    m = mu(MB.S[k]); mus.append(m)
                    if not equiv(MB.S[k]["ans"], h["ans"]):
                        dis += 1; dmu.append(m)
    if not mus:
        continue
    mus = np.array(mus)
    exs, _ = edit_examples(wl, "test")
    MB = MarginBook(wl, "test")
    ys, ms = [], []
    for e in exs:
        K = [x for _, x in e["K"]]
        if MB.has(e["subq"], K):
            ys.append(e["y"]); ms.append(MB.mu(e["subq"], K))
    res.update({"states": n, "mu_median": float(np.median(mus)), "mu_lt1": float(np.mean(mus < 1)), "mu_lt05": float(np.mean(mus < 0.5)),
                "reread_disagree": dis / n, "disagree_mu_max": float(max(dmu)) if dmu else 0.0,
                "disagree_mu_median": float(np.median(dmu)) if dmu else 0.0,
                "disagree_rate_mu_lt1": float(np.mean(np.array(dmu) < 1)) if dmu else 0.0,
                "auc_margin_only": auc(-np.array(ms), ys) if ys else 0.5, "n_test_edits": len(ys),
                "defeat_rate_mu_lt1": float(np.mean(np.array(ys)[np.array(ms) < 1])) if ys and (np.array(ms) < 1).any() else 0.0,
                "defeat_rate_mu_ge1": float(np.mean(np.array(ys)[np.array(ms) >= 1])) if ys else 0.0})
    out[wl] = res
    print(wl, res, flush=True)
jdump(out, f"{RUNS}/margin_stats.json")
