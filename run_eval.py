"""Main experiment driver: calibrate every scorer on the cal stream (conformal miss control at level alpha),
replay all policies on the test stream, run alpha sweeps and threshold sweeps (Pareto)."""
import argparse
import numpy as np
from common import *
from simulate import *
from deft import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--alpha", type=float, default=0.05)
ap.add_argument("--models", default="svdef,svde,svdef_noaux,bienc,cross")
ap.add_argument("--tag", default="")
ap.add_argument("--qpb", type=float, default=0.5, help="queries per batch as a fraction of #cached questions")
ap.add_argument("--skip_nli", action="store_true")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/eval{args.tag}"
os.makedirs(OUT, exist_ok=True)

Oc, Ot = Oracle(args.wl, "cal"), Oracle(args.wl, "test")
stream = zipf_stream(Ot.cids, Ot.nB, max(1, int(args.qpb * len(Ot.cids))))
enc = Encoder("cuda:0")
qsim_c = {c: e.float() for c, e in zip(Oc.cids, enc.encode([Oc.case[c]["q"] for c in Oc.cids], query=True))}
qsim_t = {c: e.float() for c, e in zip(Ot.cids, enc.encode([Ot.case[c]["q"] for c in Ot.cids], query=True))}
del enc
R = {"meta": {"wl": args.wl, "alpha": args.alpha, "n_cases": len(Ot.cids), "n_batches": Ot.nB,
              "n_updates": sum(len(lg["events"]) for lg in Ot.logs),
              "defeated_test": int(sum(sum(lg["changed"].values()) for lg in Ot.logs)),
              "reach_test": int(sum(len(lg["reach"]) for lg in Ot.logs)),
              "noise": Ot.R["noise"], "timing": Ot.R["timing"]}}
print(R["meta"], flush=True)


def log(name, res):
    R[name] = res
    print(f"{name:28s} SRR {res['SRR']:.4f} hit {res['hit_rate']:.3f} llm/upd {res['llm_per_update']:.2f} "
          f"P {res['inval_P']:.3f} R {res['inval_R']:.3f} miss1 {res.get('miss_test', float('nan')):.3f} "
          f"write {res['write_ms_per_batch']:.1f}ms", flush=True)
    jdump(R, f"{OUT}/results.json")


# ---------------- rule / structural policies
log("no_invalidation", replay(Ot, "none", stream))
for T in (2, 5, 10):
    log(f"ttl_{T}", replay(Ot, "ttl", stream, ttl=T))
log("provenance", replay(Ot, "provenance", stream))
log("verify_on_hit", replay(Ot, "verify_on_hit", stream, verify=Verifier(Ot)))
log("entity_rule", replay(Ot, "entity", stream))
log("trace_exact", replay(Ot, "trace_exact", stream))
R.setdefault("kinds", {})["trace_exact"] = kind_breakdown(Ot, None, 0.0)
log("eager_all", replay(Ot, "eager_all", stream))


# ---------------- calibrated scorers
def calibrated(name, mk, use_reach=True, qsim=False, alphas=(args.alpha,), sweep=True):
    sc_c, sc_t = mk(Oc), mk(Ot)
    cb = defeat_scores_batched(Oc, sc_c, use_reach, qsim_c if qsim else None)
    cs = np.concatenate(cb) if cb else np.zeros(0)
    tb = defeat_scores_batched(Ot, sc_t, use_reach, qsim_t if qsim else None)
    ts = np.concatenate(tb) if tb else np.zeros(0)
    for a in alphas:
        lam = crc_threshold(cb, a)
        res = replay(Ot, "qsim" if qsim else "learned", stream, scorer=sc_t, lam=lam, use_reach=use_reach,
                     qsim=qsim_t if qsim else None)
        res.update({"miss_batch": float(np.mean([np.mean(x < lam) for x in tb])) if tb else 0.0, "n_cal_batches": len(cb), "lambda": lam, "alpha": a, "miss_test": float(np.mean(ts < lam)) if len(ts) else 0.0,
                    "n_cal_defeats": len(cs), "n_test_defeats": len(ts), "score_ms": (1000 * sc_t.time / Ot.nB) if sc_t is not None else 0.0})
        log(name if a == args.alpha else f"{name}@a={a}", res)
        if a == args.alpha and use_reach and not qsim:
            R.setdefault("kinds", {})[name] = kind_breakdown(Ot, sc_t, lam)
    if sweep:
        fin = np.concatenate([cs[np.isfinite(cs)], ts[np.isfinite(ts)]])
        if len(fin):
            qs = np.unique(np.quantile(fin, [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]))
            curve = []
            for lam in qs:
                res = replay(Ot, "qsim" if qsim else "learned", stream, scorer=sc_t, lam=float(lam), use_reach=use_reach,
                             qsim=qsim_t if qsim else None)
                curve.append({"lambda": float(lam), "SRR": res["SRR"], "llm_per_update": res["llm_per_update"],
                              "hit_rate": res["hit_rate"], "inval_R": res["inval_R"], "miss_test": float(np.mean(ts < lam))})
            R.setdefault("pareto", {})[name] = curve
            jdump(R, f"{OUT}/results.json")


calibrated("qsim_reverse", lambda O: None, use_reach=False, qsim=True)
calibrated("reach_margin", lambda O: Scorer(O, "margin"))
if not args.skip_nli:
    nli = NLI()
    calibrated("nli_verify", lambda O: Scorer(O, "nli", nli=nli))
    del nli
models = {}
for m in args.models.split(","):
    if os.path.exists(f"{RUNS}/models/{m}{args.tag}.pt"):
        models[m] = load_model(m, args.tag)
for m, model in models.items():
    alphas = (args.alpha, 0.01, 0.02, 0.1, 0.2) if m == "svdef" else (args.alpha,)
    calibrated({"svdef": "DEFT", "svde": "DEFT-nofrag", "svdef_noaux": "DEFT-noaux", "bienc": "DEFT-bienc", "cross": "DEFT-cross"}[m],
               lambda O, model=model, m=m: Scorer(O, m, model=model), alphas=alphas)
    if m == "svdef":
        calibrated("DEFT-noreach", lambda O, model=model: Scorer(O, "svdef", model=model), use_reach=False, sweep=False)
print("done", flush=True)
