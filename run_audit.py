"""Closed-loop online calibration without an oracle (certify-to-calibrate).

In deployment nobody knows whether an entry the filter *passed* was defeated.  DEFT closes the loop with its own
exact repair operator: every flagged entry is certified anyway (its label is a by-product), and with probability pi
an update batch is *audited*, i.e. its passed touched entries are certified too.  Audited batches are a uniform random
subsample of batches, so under batch exchangeability they are exchangeable with every future batch and batch-level
CRC on them keeps its guarantee; the price is the extra certification reads, which we charge to the policy.

Per scorer and workload this script stores, for every update batch: scores of defeated / non-defeated touched entries
and the exact certification cost (reader calls) of every touched entry, then runs the time-ordered audited loop for
several pi and audit seeds.  Output: runs/<wl>/audit/results.json
"""
import argparse
import numpy as np
from common import *
from simulate import *
from deft import load_model
from deltar import load_dr

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--scorers", default="margin,nli,pair-cross,DR_rr,set-cross1_m3")
ap.add_argument("--out", default="audit")
ap.add_argument("--seeds", type=int, default=50)
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/{args.out}"
os.makedirs(OUT, exist_ok=True)
Oc, Ot = Oracle(args.wl, "cal"), Oracle(args.wl, "test")


def batches(O, scorer):
    """per batch: (defeated scores, non-defeated scores, cert cost of defeated, cert cost of non-defeated)"""
    out = []
    for b in range(1, O.nB + 1):
        held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
        chg = O.logs[b - 1]["changed"]
        sc = entry_scores(O, held, b, scorer)
        cost = {c: certify_cost(O, c, held[c], O.cur_vidx(c, b))[0] for c in O.logs[b - 1]["reach"]}
        T = list(O.logs[b - 1]["reach"])
        D = [c for c in T if chg.get(c)]; N = [c for c in T if not chg.get(c)]
        out.append((np.array([sc[c] for c in D]), np.array([sc[c] for c in N]),
                    np.array([cost[c] for c in D], dtype=float), np.array([cost[c] for c in N], dtype=float)))
    return out


def loop(cal, test, alpha, pi, seed):
    """time-ordered: batch t uses the CRC threshold of (offline calibration stream + audited earlier test batches).
    Cost per batch = certification of flagged entries (+ of passed entries if the batch is audited)."""
    rng = np.random.default_rng(seed)
    hist = [d for d, _, _, _ in cal if len(d)]
    risks, saved, ntot, calls, audit_calls, n_aud = [], 0, 0, 0.0, 0.0, 0
    for d, n, cd, cn in test:
        lam = crc_threshold(hist, alpha)
        fd, fn = d >= lam, n >= lam
        calls += cd[fd].sum() + cn[fn].sum()
        if len(d):
            risks.append(float(np.mean(~fd)))
        saved += int((~fn).sum()); ntot += len(n)
        if rng.random() < pi:                      # audit: certify the passed entries too -> exact labels
            n_aud += 1
            audit_calls += cd[~fd].sum() + cn[~fn].sum()
            if len(d):
                hist.append(d)
    nb = len(test)
    return {"risk": float(np.mean(risks)) if risks else 0.0, "save": saved / max(ntot, 1),
            "calls_per_batch": calls / nb, "audit_calls_per_batch": audit_calls / nb, "audited": n_aud}


def exact_calls(test):
    return float(np.mean([cd.sum() + cn.sum() for _, _, cd, cn in test]))


def mk(name):
    if name == "margin":
        return lambda O: Scorer(O, "margin")
    if name == "nli":
        nli = NLI()
        return lambda O: Scorer(O, "nli", nli=nli)
    if name == "pair-cross":
        model = load_model("cross")
        return lambda O: Scorer(O, "cross", model=model)
    if name == "DR_rr":
        model = load_dr("full", "_rr")
        return lambda O: SetScorer(O, model)
    if name.startswith("set-cross1"):
        model = load_dr("setcross1", name[len("set-cross1"):])
        return lambda O: SetScorer(O, model)
    raise ValueError(name)


R = {"meta": {"wl": args.wl}}
first = True
for name in [x for x in args.scorers.split(",") if x]:
    try:
        f = mk(name)
    except Exception as e:
        print("skip", name, e, flush=True); continue
    cal, test = batches(Oc, f(Oc)), batches(Ot, f(Ot))
    import pickle
    pickle.dump({"cal": cal, "test": test}, open(f"{OUT}/arr_{name}.pkl", "wb"))
    print("dumped", name, flush=True)
    if not os.environ.get("AUDIT_LOOP"):          # the controllers are evaluated offline by analyze_audit.py
        continue
    if first:
        R["meta"]["exact_cert_calls_per_batch"] = exact_calls(test); R["meta"]["n_test_batches"] = len(test); first = False
    res = {}
    for a in (0.05, 0.1, 0.2):
        for pi in (0.0, 0.1, 0.25, 0.5, 1.0):
            runs = [loop(cal, test, a, pi, s) for s in range(args.seeds if 0 < pi < 1 else 1)]
            res[f"{a}|{pi}"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            res[f"{a}|{pi}"]["risk_std"] = float(np.std([r["risk"] for r in runs]))
            print(name, a, pi, {k: round(v, 4) for k, v in res[f"{a}|{pi}"].items()}, flush=True)
    R[name] = res
    jdump(R, f"{OUT}/results.json")
    torch.cuda.empty_cache()
print("done", flush=True)
