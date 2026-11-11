"""v3 evaluation: Delta-Reader variants + re-scored v2 filters, on top of the baseline results of run_eval.py.

Adds per scorer
  * replay at the target alpha (fixed cal/test streams) and an alpha sweep,
  * split validity: cal and test batches are pooled and re-split at random (exchangeable batch units); CRC is
    re-calibrated on each half and evaluated on the other.  Reports realised batch miss risk (mean, std, q90) and
    the saving = fraction of touched-but-not-defeated entries that are *not* recomputed.
"""
import argparse, shutil
import numpy as np
from common import *
from simulate import *
from deft import load_model
from deltar import load_dr

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--dr", default="full,noset,doclevel,noaux,setcross")
ap.add_argument("--v2", default="svdef,cross")
ap.add_argument("--tag", default="")
ap.add_argument("--out", default="eval3")
ap.add_argument("--qpb", type=float, default=0.5)
ap.add_argument("--splits", type=int, default=200)
ap.add_argument("--skip_nli", action="store_true")
ap.add_argument("--extra", action="store_true", help="recent baselines: reader-judge, FreshCache-style, set-level NLI")
ap.add_argument("--skip_margin", action="store_true")
ap.add_argument("--spec", default="", help="speculators: comma list of <model>[:<adapter>][:<mode>] (spec.py)")
ap.add_argument("--splits_only", action="store_true", help="skip the trace_exact+cert / margin / nli / v2 / dr parts")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/{args.out}"
os.makedirs(OUT, exist_ok=True)
base = f"{RUNS}/{args.wl}/eval/results.json"
R = jload(base) if os.path.exists(base) else {}
R.setdefault("meta", {})["v3"] = True
R["split"] = {}

Oc, Ot = Oracle(args.wl, "cal"), Oracle(args.wl, "test")
stream = zipf_stream(Ot.cids, Ot.nB, max(1, int(args.qpb * len(Ot.cids))))


def log(name, res):
    R[name] = res
    print(f"{name:26s} SRR {res['SRR']:.4f} hit {res['hit_rate']:.3f} llm/upd {res['llm_per_update']:.2f} "
          f"P {res['inval_P']:.3f} R {res['inval_R']:.3f} mb {res.get('miss_batch', float('nan')):.3f} "
          f"write {res['write_ms_per_batch']:.1f}ms", flush=True)
    jdump(R, f"{OUT}/results.json")


def touched_batched(O, scorer):
    """per batch with at least one touched entry: (scores of defeated entries, scores of touched non-defeated)"""
    out = []
    for b in range(1, O.nB + 1):
        held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
        chg = O.logs[b - 1]["changed"]
        sc = entry_scores(O, held, b, scorer)
        touched = [c for c in O.logs[b - 1]["reach"]]
        d = np.array([sc[c] for c in touched if chg.get(c)]); n = np.array([sc[c] for c in touched if not chg.get(c)])
        out.append((d, n))
    return out


def entry_kind_breakdown(O, scorer, lam):
    """entry level, for touched entries with a single cause kind: #entries, #defeated, #flagged (score >= lam)."""
    out = collections.defaultdict(collections.Counter)
    for b in range(1, O.nB + 1):
        lg = O.logs[b - 1]
        held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
        pairs, _ = reach_pairs(O, held, b)
        cause = collections.defaultdict(set)
        for (c, v, h), r in pairs:
            cause[c].add(O.ins_kind.get(r, "?"))
        kind_of_doc = {d: k for op, k, d in lg["events"] if op in ("delete", "modify")}
        rem = set(lg["rem"])
        for c, v in held.items():
            for hop in O.trace(c, v):
                for x in hop["rows"]:
                    if x in rem:
                        k = kind_of_doc.get(O.doc_of_row[x], "?")
                        cause[c].add(k if k.startswith("delete") else k + "(old)")
        sc = entry_scores(O, held, b, scorer)
        for c in lg["reach"]:
            ks = cause.get(c, set())
            if len(ks) != 1:
                continue
            k = next(iter(ks))
            out[k]["n"] += 1; out[k]["defeated"] += int(bool(lg["changed"].get(c))); out[k]["flagged"] += int(sc[c] >= lam)
            out[k]["fp_flag"] += int(sc[c] >= lam and not lg["changed"].get(c))
            out[k]["fn_miss"] += int(sc[c] < lam and bool(lg["changed"].get(c)))
    return {k: dict(v) for k, v in out.items()}


def temporal_validity(cal, test, alpha):
    """no shuffling: (i) calibrate on the calibration stream, evaluate on the test stream in four chronological
    quarters; (ii) temporal hold-out: calibrate on calibration + first half of the test stream, evaluate on the
    second half."""
    def ev(C, T):
        lam = crc_threshold([d for d, _ in C if len(d)], alpha)
        Td = [d for d, _ in T if len(d)]
        nn_ = np.concatenate([x for _, x in T]) if T else np.zeros(0)
        return {"risk": float(np.mean([np.mean(d < lam) for d in Td])) if Td else 0.0,
                "save": float(np.mean(nn_ < lam)) if len(nn_) else 0.0}
    q = np.array_split(np.arange(len(test)), 4)
    h = len(test) // 2
    return {"quarters": [ev(cal, [test[i] for i in ix]) for ix in q], "holdout": ev(cal + test[:h], test[h:])}


def prequential(cal, test, alpha):
    """online, time-ordered calibration: test batch t is judged with the threshold calibrated on the calibration
    stream and all test batches before t (no shuffling, no look-ahead)."""
    hist = [d for d, _ in cal if len(d)]
    risks, saves = [], []
    for d, n in test:
        lam = crc_threshold(hist, alpha)
        if len(d):
            risks.append(float(np.mean(d < lam))); hist.append(d)
        if len(n):
            saves.append(n < lam)
    sv_ = np.concatenate(saves) if saves else np.zeros(0)
    q = np.array_split(np.arange(len(risks)), 4)
    return {"risk": float(np.mean(risks)) if risks else 0.0, "save": float(sv_.mean()) if len(sv_) else 0.0,
            "risk_quarters": [float(np.mean([risks[i] for i in ix])) if len(ix) else 0.0 for ix in q]}


def fit_freshcache(O):
    """FreshCache-style staleness model fitted on the calibration stream: per-entry hazard of an answer change as a
    function of the age since the last change; exponential (constant hazard) and an MLP over (age, #hops)."""
    from sklearn.neural_network import MLPClassifier
    X, y = [], []
    for c in O.cids:
        last = 0
        for b in range(1, O.nB + 1):
            ch = bool(O.logs[b - 1]["changed"].get(c))
            X.append((b - last, len(O.trace(c, O.cur_vidx(c, b - 1))))); y.append(int(ch))
            if ch:
                last = b
    X, y = np.array(X, dtype=float), np.array(y)
    lam = y.mean()
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(X / [O.nB, 4.0], y)
    cache = {}
    def p_mlp(a, m):
        if (a, m) not in cache:
            hz = mlp.predict_proba(np.array([[i, m] for i in range(1, a + 1)], dtype=float) / [O.nB, 4.0])[:, 1] if a > 0 else np.zeros(0)
            cache[(a, m)] = 1 - float(np.prod(1 - hz))
        return cache[(a, m)]
    return (lambda a, m: 1 - (1 - lam) ** a), p_mlp


def split_validity(cal, test, alpha, n=200):
    pool = cal + test
    rng = np.random.default_rng(0)
    risks, saves = [], []
    for s in range(n):
        perm = rng.permutation(len(pool)); h = len(pool) // 2
        A = [pool[i] for i in perm[:h]]; B = [pool[i] for i in perm[h:]]
        lam = crc_threshold([d for d, _ in A if len(d)], alpha)
        Bd = [d for d, _ in B if len(d)]
        risks.append(np.mean([np.mean(d < lam) for d in Bd]) if Bd else 0.0)
        nn_ = np.concatenate([x for _, x in B]) if B else np.zeros(0)
        saves.append(float(np.mean(nn_ < lam)) if len(nn_) else 0.0)
    risks, saves = np.array(risks), np.array(saves)
    return {"risk_mean": float(risks.mean()), "risk_std": float(risks.std()), "risk_q90": float(np.quantile(risks, 0.9)),
            "save_mean": float(saves.mean()), "save_std": float(saves.std())}


def calibrated(name, mk, alphas=(args.alpha, 0.01, 0.02, 0.1, 0.2)):
    sc_c, sc_t = mk(Oc), mk(Ot)
    cal, test = touched_batched(Oc, sc_c), touched_batched(Ot, sc_t)
    cb = [d for d, _ in cal if len(d)]; tb = [d for d, _ in test if len(d)]
    for a in alphas:
        lam = crc_threshold(cb, a)
        res = replay(Ot, "learned", stream, scorer=sc_t, lam=lam)
        tn = np.concatenate([x for _, x in test])
        res.update({"miss_batch": float(np.mean([np.mean(x < lam) for x in tb])) if tb else 0.0, "n_cal_batches": len(cb),
                    "lambda": lam, "alpha": a, "save_test": float(np.mean(tn < lam)) if len(tn) else 0.0,
                    "score_ms": 1000 * sc_t.time / Ot.nB})
        log(name if a == args.alpha else f"{name}@a={a}", res)
        if a == args.alpha:
            rc = replay(Ot, "learned", stream, scorer=sc_t, lam=lam, certify=True)
            rc.update({"alpha": a, "lambda": lam}); log(name + "+cert", rc)
    R["split"][name] = {str(a): split_validity(cal, test, a, args.splits) for a in (0.02, 0.05, 0.1, 0.2)}
    R.setdefault("temporal", {})[name] = {str(a): temporal_validity(cal, test, a) for a in (0.05, 0.1, 0.2)}
    R.setdefault("prequential", {})[name] = {str(a): prequential(cal, test, a) for a in (0.05, 0.1, 0.2)}
    print("  split", name, R["split"][name][str(args.alpha)], flush=True)
    R.setdefault("ekinds", {})[name] = entry_kind_breakdown(Ot, sc_t, crc_threshold(cb, args.alpha))
    jdump(R, f"{OUT}/results.json")


if args.spec:
    from spec import Speculator
    from spec_scorer import SpecScorer
    for item in args.spec.split(","):
        parts = item.split(":")
        MODES = ("zs", "did", "ll", "shift", "judge", "reans")
        adapter = parts[1] if len(parts) > 1 and parts[1] not in MODES else ""
        mode = next((p for p in parts[1:] if p in MODES), "zs")
        sp = Speculator(parts[0], adapter)
        calibrated(f"SPEC-{item}", lambda O, sp=sp, mode=mode: SpecScorer(O, sp, mode))
        del sp; torch.cuda.empty_cache()
    if args.splits_only:
        print("done", flush=True); raise SystemExit
log("trace_exact+cert", replay(Ot, "trace_exact", stream, certify=True))
if args.extra:
    log("judge_on_hit", replay(Ot, "judge_on_hit", stream))
    f_exp, f_mlp = fit_freshcache(Oc)
    for eps in (0.05, 0.1):
        log(f"freshcache_exp@{eps}", replay(Ot, "freshcache", stream, lam=eps, fc=f_exp))
        log(f"freshcache_mlp@{eps}", replay(Ot, "freshcache", stream, lam=eps, fc=f_mlp))
R.setdefault("ekinds", {})["trace_exact"] = entry_kind_breakdown(Ot, None, -np.inf)
if not args.skip_margin:
    calibrated("reach_margin", lambda O: Scorer(O, "margin"))
if not args.skip_nli:
    nli = NLI()
    calibrated("nli_verify", lambda O: Scorer(O, "nli", nli=nli))
    if args.extra:
        calibrated("set_nli", lambda O: SetNLIScorer(O, nli))
    del nli; torch.cuda.empty_cache()
for m in [x for x in args.v2.split(",") if x]:
    if os.path.exists(f"{RUNS}/models/{m}.pt"):
        model = load_model(m)
        calibrated({"svdef": "DEFT-v2", "cross": "pair-cross"}[m], lambda O, model=model, m=m: Scorer(O, m, model=model))
        del model; torch.cuda.empty_cache()
names = {"full": "DR", "noset": "DR-noset", "doclevel": "DR-doclevel", "noaux": "DR-noaux", "setcross": "set-cross", "setcross2": "set-cross2", "setcross1": "set-cross1", "setcrossE": "enter-cross", "setcrossD": "delta-cross", "mtr": "MTR", "mtr0": "MTR-nomu", "mtrE": "MTR-edit"}
for item in [x for x in args.dr.split(",") if x]:
    v, tag = (item.split(":") + [args.tag])[:2]
    if os.path.exists(f"{RUNS}/models/dr_{v}{tag}.pt"):
        model = load_dr(v, tag)
        calibrated(names[v] + tag, lambda O, model=model: SetScorer(O, model))
        del model; torch.cuda.empty_cache()
print("done", flush=True)
