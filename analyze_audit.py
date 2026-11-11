"""Offline analysis of the closed-loop (oracle-free) online defeat control, from runs/<wl>/audit/arr_<scorer>.pkl.

Two controllers, both fed ONLY by labels that DEFT's own certification produces:
  * flagged entries are certified anyway -> their labels are free;
  * with probability pi a batch is audited: its passed touched entries are certified too (cost charged).
CRC-audit : threshold = batch-level CRC on (offline calibration stream + audited earlier batches).
            Guarantee in expectation under batch exchangeability (audits are independent coin flips).
ACI-audit : the CRC level itself is adapted online on audited batches,  a_{t+1} = a_t + g (alpha - L_t),
            which bounds the long-run average miss loss over audited batches deterministically, for ANY sequence of
            batches and cache states (no exchangeability):  |mean L - alpha| <= (1 + g) / (g T_audit).
Usage: python analyze_audit.py <results_dir_with_wl_subdirs> <out.json>
"""
import sys, os, json, pickle, glob
import numpy as np

SEEDS = 50
GAMMA = 0.05


def crc(batches, alpha):
    n = len(batches)
    if n == 0 or alpha <= 0 or 1.0 / (n + 1) > alpha:
        return -np.inf
    if alpha >= 1:
        return np.inf
    cand = np.unique(np.concatenate([x[np.isfinite(x)] for x in batches] + [np.array([-np.inf])]))
    risk = np.zeros(len(cand))
    for x in batches:                                  # risk(lam) = mean_i |{s in D_i : s < lam}| / |D_i|
        risk += np.searchsorted(np.sort(x), cand, side="left") / len(x)
    ok = n / (n + 1) * risk / n + 1.0 / (n + 1) <= alpha
    bad = np.nonzero(~ok)[0]
    last = (bad[0] - 1) if len(bad) else len(cand) - 1
    return float(cand[last]) if last >= 0 else -np.inf


def run(cal, test, alpha, pi, seed, mode):
    rng = np.random.default_rng(seed)
    hist = [d for d, _, _, _ in cal if len(d)]
    a_t = alpha
    risks, aud_risks, saved, ntot, calls, acalls, naud = [], [], 0, 0, 0.0, 0.0, 0
    for d, n, cd, cn in test:
        lam = crc(hist, a_t if mode == "aci" else alpha)
        fd, fn = d >= lam, n >= lam
        calls += cd[fd].sum() + cn[fn].sum()
        L = float(np.mean(~fd)) if len(d) else None
        if L is not None:
            risks.append(L)
        saved += int((~fn).sum()); ntot += len(n)
        if rng.random() < pi:
            naud += 1
            acalls += cd[~fd].sum() + cn[~fn].sum()
            if L is not None:
                hist.append(d); aud_risks.append(L)
                if mode == "aci":
                    a_t = a_t + GAMMA * (alpha - L)
    nb = len(test)
    q = np.array_split(np.arange(len(risks)), 4)
    return {"risk": float(np.mean(risks)) if risks else 0.0,
            "risk_q4_max": float(max(np.mean([risks[i] for i in ix]) for ix in q if len(ix))) if risks else 0.0,
            "save": saved / max(ntot, 1), "calls": calls / nb, "audit_calls": acalls / nb, "audited": naud}


def summarise(cal, test):
    out = {"exact_cert_calls": float(np.mean([cd.sum() + cn.sum() for _, _, cd, cn in test]))}
    for mode in ("crc", "aci"):
        for a in (0.05, 0.1, 0.2):
            for pi in (0.1, 0.25, 0.5, 1.0):
                rs = [run(cal, test, a, pi, s, mode) for s in range(SEEDS if pi < 1 else 1)]
                r = {k: float(np.mean([x[k] for x in rs])) for k in rs[0]}
                r["risk_std"] = float(np.std([x["risk"] for x in rs]))
                out[f"{mode}|{a}|{pi}"] = r
    return out


def job(f):
    wl = os.path.basename(os.path.dirname(f)); sc = os.path.basename(f)[4:-4]
    D = pickle.load(open(f, "rb"))
    return wl, sc, summarise(D["cal"], D["test"])


if __name__ == "__main__":
    import multiprocessing as mp
    root, outp = sys.argv[1], sys.argv[2]
    R = json.load(open(outp)) if os.path.exists(outp) else {}
    todo = [f for f in sorted(glob.glob(f"{root}/*/arr_*.pkl"))
            if os.path.basename(f)[4:-4] not in R.get(os.path.basename(os.path.dirname(f)), {})]
    with mp.Pool(min(len(todo), max(1, os.cpu_count() - 2)) or 1) as pool:
        for wl, sc, x in pool.imap_unordered(job, todo):
            R.setdefault(wl, {})[sc] = x
            print(wl, sc, "exact", round(x["exact_cert_calls"], 2),
                  {m: (round(100 * x[f"{m}|0.1|0.25"]["risk"], 1), round(100 * x[f"{m}|0.1|0.25"]["save"], 1),
                       round(x[f"{m}|0.1|0.25"]["calls"] + x[f"{m}|0.1|0.25"]["audit_calls"], 2)) for m in ("crc", "aci")}, flush=True)
            json.dump(R, open(outp, "w"), indent=1)
