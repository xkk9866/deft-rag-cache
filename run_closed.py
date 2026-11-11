"""Closed-loop replay of the deployed system with oracle-free online control, reporting three distinct risks.

Per update batch t (strict arrival order, cache state carried across batches exactly as in simulate.replay):
  1. score the touched entries of the CURRENT cache state;  threshold lambda_t = CRC at working level a_t on the
     offline calibration stream plus the audited test batches so far;
  2. flagged entries become suspects (certified on their next request, hop-locally);
  3. with probability pi (independent coin Z_t) the batch is audited: every touched entry is certified now, which
     reveals the batch loss L_t and repairs the misses of this batch;
  4. a_{t+1} = a_t + (gamma / pi) Z_t (alpha - L_t)   (importance-weighted adaptive control; Theorem in the paper).
Risks: filter miss  = mean_t L_t over defeating batches (before any repair);
       residual miss = mean_t (1 - Z_t) L_t (defeated answers that are still cached after the batch is processed);
       SRR           = share of requests served a stale answer (request level, with popularity and dwell time).
Cost: reader calls per update including flagged certifications, audit certifications and recomputations.
Output: runs/<wl>/closed/results.json
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
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--gamma", type=float, default=0.05)
ap.add_argument("--hist_from", default="same", help="offline calibration: same | none (cold start) | <other workload>")
ap.add_argument("--alphas", default="0.05,0.1,0.2")
ap.add_argument("--pis", default="0.1,0.25")
ap.add_argument("--out", default="results")
ap.add_argument("--sweep", type=int, default=0, help="every SWEEP batches, certify every cached entry with unverified hops (bounds stale lifetime)")
ap.add_argument("--dump", action="store_true", help="dump per-request work logs (exact and DEFT-X, alpha 0.1, pi 0.25, seed 0) for serve_bench.py")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/closed"; os.makedirs(OUT, exist_ok=True)
Ot = Oracle(args.wl, "test")
Oc = None if args.hist_from == "none" else Oracle(args.wl if args.hist_from == "same" else args.hist_from, "cal")
stream = zipf_stream(Ot.cids, Ot.nB, max(1, int(0.5 * len(Ot.cids))))


def crc(batches, a):
    n = len(batches)
    if n == 0 or a <= 0 or 1.0 / (n + 1) > a:
        return -np.inf
    if a >= 1:
        return np.inf
    cand = np.unique(np.concatenate([x[np.isfinite(x)] for x in batches] + [np.array([-np.inf])]))
    risk = np.zeros(len(cand))
    for x in batches:
        risk += np.searchsorted(np.sort(x), cand, side="left") / len(x)
    ok = risk / (n + 1) + 1.0 / (n + 1) <= a
    bad = np.nonzero(~ok)[0]
    last = (bad[0] - 1) if len(bad) else len(cand) - 1
    return float(cand[last]) if last >= 0 else -np.inf


def cal_batches(O, scorer):
    out = []
    for b in range(1, O.nB + 1):
        held = {c: O.cur_vidx(c, b - 1) for c in O.cids}
        sc = entry_scores(O, held, b, scorer)
        d = np.array([sc[c] for c in O.logs[b - 1]["reach"] if O.logs[b - 1]["changed"].get(c)])
        if len(d):
            out.append(d)
    return out


def closed_loop(scorer, hist0, alpha, pi, seed, exact=False, log=None):
    rng = np.random.default_rng(seed)
    hist = list(hist0); a_t = alpha
    held = {c: 0 for c in Ot.cids}; suspect = {}
    m = collections.Counter(); L, Lres = [], []
    t_score = scorer.time if hasattr(scorer, "time") else 0.0
    ep = {}; episodes = []                               # stale episodes: cid -> [start batch, stale serves]
    for b in range(1, Ot.nB + 1):
        cur = {c: Ot.cur_vidx(c, b) for c in Ot.cids}
        live = {c: v for c, v in held.items() if v is not None}
        sc = entry_scores(Ot, live, b, scorer)
        touched = [c for c, s in sc.items() if s > -np.inf]
        lam = -np.inf if exact else crc(hist, a_t)
        flagged = {c for c in touched if sc[c] >= lam}
        defeated = [c for c in touched if not equiv(Ot.ans(c, live[c]), Ot.ans(c, cur[c]))]
        Z = (not exact) and rng.random() < pi
        if defeated:
            Lt = float(np.mean([c not in flagged for c in defeated]))
            L.append(Lt); Lres.append(0.0 if Z else Lt)
        for c in flagged:
            suspect[c] = held[c]; held[c] = None
        if Z:                                              # audit: certify every touched entry now (exact labels)
            for c in touched:
                v0 = suspect.pop(c, None) if c in flagged else held[c]
                if v0 is None:
                    continue
                cost, _ = certify_cost(Ot, c, Ot.vi(v0), cur[c]); m["llm"] += cost; m["audit"] += cost
                if log is not None:
                    log.append((b, "audit", c, Ot.vi(v0), cur[c]))
                held[c] = cur[c]
            if defeated:
                hist.append(np.array([sc[c] for c in defeated]))
                a_t = a_t + args.gamma * (alpha - Lt)      # = (gamma'/pi) Z_t (alpha - L_t) with gamma' = gamma * pi
        if args.sweep and b % args.sweep == 0:            # periodic sweep: certify entries touched since their trusted computation
            for c, v in held.items():
                if v is not None and Ot.vi(v) != cur[c]:
                    cost, _ = certify_cost(Ot, c, Ot.vi(v), cur[c]); m["llm"] += cost; m["sweep"] += cost; held[c] = cur[c]
        held = refresh_states(Ot, held, b, [c for c, v in held.items() if v is not None])
        for c, v in held.items():                         # open / close stale episodes (entry cached but no longer equal to recomputation)
            st_ = v is not None and not equiv(Ot.ans(c, v), Ot.ans(c, cur[c]))
            if st_ and c not in ep:
                ep[c] = [b, 0]
            elif not st_ and c in ep:
                s0, n0 = ep.pop(c); episodes.append((b - s0, n0, False))
        for c in stream[b - 1]:
            m["q"] += 1
            if held[c] is None and c in suspect:
                v0 = Ot.vi(suspect.pop(c))
                cost, ok = certify_cost(Ot, c, v0, cur[c]); held[c] = cur[c]; m["llm"] += cost
                if log is not None:
                    log.append((b, "cert", c, v0, cur[c]))
                continue
            if held[c] is None:
                held[c] = cur[c]; m["llm"] += Ot.calls(c, cur[c])
                if log is not None:
                    log.append((b, "miss", c, None, cur[c]))
                continue
            if log is not None:
                log.append((b, "hit", c, None, None))
            if not equiv(Ot.ans(c, held[c]), Ot.ans(c, cur[c])):
                m["stale"] += 1
                if c in ep:
                    ep[c][1] += 1
        for c in list(ep):                                 # a served stale entry that was recomputed on a miss closes too
            if held[c] is None or equiv(Ot.ans(c, held[c]), Ot.ans(c, cur[c])):
                s0, n0 = ep.pop(c); episodes.append((b - s0, n0, False))
    for c, (s0, n0) in ep.items():
        episodes.append((Ot.nB + 1 - s0, n0, True))      # still stale when the stream ends (right-censored)
    n_upd = sum(len(lg["events"]) for lg in Ot.logs)
    q = np.array_split(np.arange(len(L)), 4)
    T = len(L); Ta = int(round(pi * T)); delta = 0.05
    lens = np.array([e[0] for e in episodes]) if episodes else np.zeros(1)
    serves = np.array([e[1] for e in episodes]) if episodes else np.zeros(1)
    return {"filter_miss": float(np.mean(L)) if L else 0.0, "residual_miss": float(np.mean(Lres)) if Lres else 0.0,
            "SRR": m["stale"] / m["q"], "calls_per_update": m["llm"] / n_upd, "audit_per_update": m["audit"] / n_upd,
            "sweep_per_update": m["sweep"] / n_upd, "filter_ms_per_update": 1000 * ((scorer.time if hasattr(scorer, "time") else 0.0) - t_score) / n_upd,
            "fm_q1": float(np.mean([L[i] for i in q[0]])) if len(q[0]) else 0.0, "fm_q4": float(np.mean([L[i] for i in q[-1]])) if len(q[-1]) else 0.0,
            "T_defeating": T, "bound": (1 + 2 * args.gamma) / (args.gamma * max(T, 1)) + np.sqrt(2 * np.log(2 / delta) / max(T, 1)) / pi,
            "episodes": len(episodes), "ep_len_mean": float(lens.mean()), "ep_len_p95": float(np.percentile(lens, 95)),
            "ep_serves_mean": float(serves.mean()), "ep_serves_p95": float(np.percentile(serves, 95)),
            "censored": int(sum(e[2] for e in episodes))}


def mk(name):
    if name == "margin":
        return lambda O: Scorer(O, "margin")
    if name == "nli":
        nli = NLI(); return lambda O: Scorer(O, "nli", nli=nli)
    if name == "pair-cross":
        model = load_model("cross"); return lambda O: Scorer(O, "cross", model=model)
    if name == "DR_rr":
        model = load_dr("full", "_rr"); return lambda O: SetScorer(O, model)
    if name.startswith("set-cross1"):
        model = load_dr("setcross1", name[len("set-cross1"):]); return lambda O: SetScorer(O, model)
    if name.startswith("dr:"):                               # dr:<variant>:<tag>, e.g. dr:setcrossD:_m3
        _, v, tag = name.split(":"); model = load_dr(v, tag); return lambda O: SetScorer(O, model)
    raise ValueError(name)


R = {"meta": {"wl": args.wl, "gamma": args.gamma, "hist_from": args.hist_from}}
R["meta"]["exact_cert"] = replay(Ot, "trace_exact", stream, certify=True)
for name in [x for x in args.scorers.split(",") if x]:
    try:
        f = mk(name)
    except Exception as e:
        print("skip", name, e, flush=True); continue
    sc_t = f(Ot)
    hist0 = cal_batches(Oc, f(Oc)) if Oc is not None else []
    res = {}
    for a in [float(x) for x in args.alphas.split(",")]:
        for pi in [float(x) for x in args.pis.split(",")]:
            runs = [closed_loop(sc_t, hist0, a, pi, s) for s in range(args.seeds)]
            res[f"{a}|{pi}"] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            res[f"{a}|{pi}"].update({k + "_std": float(np.std([r[k] for r in runs])) for k in ("filter_miss", "SRR")})
            res[f"{a}|{pi}"]["filter_ms_per_update"] = max(r["filter_ms_per_update"] for r in runs)   # scores are memoised after the first run
            print(name, a, pi, {k: round(v, 4) for k, v in res[f"{a}|{pi}"].items()}, flush=True)
    R[name] = res
    jdump(R, f"{OUT}/{args.out}.json")
    del f, sc_t; torch.cuda.empty_cache()
if args.dump:
    import pickle
    lg_ex = []; closed_loop(None, [], 0.1, 0.0, 0, exact=True, log=lg_ex)
    f = mk("set-cross1_m3"); sc_t = f(Ot); hist0 = cal_batches(Oc, f(Oc)) if Oc is not None else []
    lg_x = []; closed_loop(sc_t, hist0, 0.1, 0.25, 0, log=lg_x)
    pickle.dump({"exact": lg_ex, "deftx": lg_x}, open(f"{OUT}/reqlog.pkl", "wb"))
    print("dumped", len(lg_ex), len(lg_x), flush=True)
print("done", flush=True)
