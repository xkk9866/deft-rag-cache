"""Request-randomised certification: the deployed DEFT system, closed loop, in arrival order.

Write path per update batch: exact trace reachability and state refresh only (Thm. 1, invariant I1).  Every hop whose
evidence changed becomes *unverified* (current K_j differs from the trusted K*_j).  No reader call and no neural
network runs on the write path.
Read path per request for entry e:
  U_e empty -> serve (consistent by trace invariance);
  otherwise -> p_e = 1 - prod_{j in U_e} (1 - p_j), p_j the calibrated defeat probability of the cumulative edit
               K*_j -> K_j (scored lazily, once per state);  rho_e = max(rho_min, 1 - a_t / p_e);
               with probability rho_e certify (hop-local, speculative, hop states shared across entries), else serve.
  controller: a_{t+1} = min(1, a_t + gamma (alpha - S_hat_t)),  S_hat_t = V_t S_t (1 - rho_t) / rho_t, an unbiased
               estimate of "request t is served stale" (S_t is revealed by the certification).
Theorem (paper): at fixed horizon N, use a geometric upper rounding w of W_N and log(J/delta), where J is the
number of variance-grid points. SRR_N <= alpha + (a_1-a_{N+1})^+/(gamma N) + (sqrt(2*w*log(J/delta))+2*log(J/delta))/N.
Here W_N = sum_t S_t (1-rho_t)/rho_t and b = (1-rho_min)/rho_min, with the actual propensity clipped to [rho_min, 1]; every stale
episode is served at most (1 - rho_min) / rho_min times in expectation.
Output: runs/<wl>/closed2/<out>.json
"""
import argparse, math, time
import numpy as np
from common import *
from simulate import *
from deltar import load_dr, score_examples, edit_examples
from mcert import MarginBook
from request_risk import certification_probability

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--scorers", default="mtr:_s0,setcross1:_aug,setcross1:_m3,nli,margin,uniform,oracle")
ap.add_argument("--alphas", default="0.005,0.01,0.02")
ap.add_argument("--rhos", default="0.05,0.1")
ap.add_argument("--gammas", default="0.01")
ap.add_argument("--seeds", type=int, default=10)
ap.add_argument("--noshare", action="store_true", help="also run without hop-state sharing (main scorer only)")
ap.add_argument("--out", default="results")
ap.add_argument("--baselines", default="", help="published systems, same closed loop: gptcache,ttl,vcache,fincache,grounded,freshcache")
ap.add_argument("--dump", default="", help="scorer whose request log (alpha 0.01, rho 0.05, seed 0) is dumped for serve_bench2")
ap.add_argument("--cache_sizes", default="", help="cache-size study: comma list of sub-cache sizes (random subsets of the cached questions)")
ap.add_argument("--size_seeds", type=int, default=3)
ap.add_argument("--reliability", action="store_true", help="record (p_e, stale) of certified requests and report calibration bins")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/closed2"; os.makedirs(OUT, exist_ok=True)
Ot = Oracle(args.wl, "test")
stream = zipf_stream(Ot.cids, Ot.nB, max(1, int(0.5 * len(Ot.cids))))
n_upd = sum(len(lg["events"]) for lg in Ot.logs)
MBt = MarginBook(args.wl, "test")
TOPK = {}


def topk_rows(subq, b):
    """exact current top-k of a sub-question after batch b (memoised: deterministic in (subq, b))"""
    key = (subq, b)
    if key not in TOPK:
        s = Ot.qemb[subq] @ Ot.E.T
        s[~Ot.alive[b]] = -1e4
        sc, top = torch.topk(s, Ot.k)
        TOPK[key] = (tuple(top.tolist()), tuple(float(x) for x in sc.tolist()))
    return TOPK[key]


def refresh(held, b):
    """invariant I1: touched hops of every cached entry take their exact post-batch evidence set"""
    live = {c: v for c, v in held.items() if v is not None}
    pairs, _ = reach_pairs(Ot, live, b)
    hops = {(c, h) for (c, v, h), r in pairs}
    rem = set(Ot.logs[b - 1]["rem"])
    if rem:
        for c, v in live.items():
            for h, hop in enumerate(Ot.trace(c, v)):
                if any(x in rem for x in hop["rows"]):
                    hops.add((c, h))
    by = collections.defaultdict(list)
    for c, h in sorted(hops):
        rows, scores = topk_rows(Ot.trace(c, held[c])[h]["subq"], b)
        by[c].append((h, rows, scores))
    for c, ov in by.items():
        v = held[c]
        prev = dict((h, (r, s_)) for h, r, s_ in v[2]) if isinstance(v, tuple) else {}
        for h, r, s_ in ov:
            prev[h] = (r, s_)
        held[c] = ("R", Ot.vi(v), tuple(sorted((h, r, s_) for h, (r, s_) in prev.items())))
    return held, set(by)


def unverified_items(c, v):
    """(subq, trusted answer, K*, K, mu*) of every hop whose current evidence differs from the trusted evidence"""
    if not isinstance(v, tuple):
        return []
    tr0 = Ot.versions[c][Ot.vi(v)][1]["trace"]
    out = []
    for h, rows, _ in v[2]:
        K0 = tuple(tr0[h]["rows"])
        if tuple(rows) != K0:
            out.append((tr0[h]["subq"], tr0[h]["ans"], K0, tuple(rows), MBt.mu(tr0[h]["subq"], K0)))
    return out


class HopTable:
    """The shared hop-state table.  The reader is a deterministic function of the hop state (sub-question, evidence
    rows), so once any entry has computed or verified G(q, K) the value is known for every entry (Prop. share).
    level 0: nothing shared;  level 1: verifications of a re-read state are shared (the earlier engine);
    level 2: in addition, the reader calls of a *recomputed suffix* reuse any hop state already in the table
    (shared computation: the cache-level analogue of multi-query optimisation).  Decomposer calls depend on the
    original question and are never shared."""

    def __init__(self, level):
        self.level, self.known = level, set()
        if level >= 2:                                               # every initially cached hop state is known
            for c in Ot.cids:
                for hop in Ot.trace(c, 0):
                    self.known.add((hop["subq"], tuple(hop["rows"])))

    def read(self, subq, rows, verify=False):
        """cost of one reader call on state (subq, rows): 0 if already known under the sharing level"""
        if self.level == 0 or (self.level == 1 and not verify):
            return 1
        key = (subq, tuple(rows))
        if key in self.known:
            return 0
        self.known.add(key)
        return 1

    def suffix(self, tn, start):
        """recompute hops start..m-1 of the new trace: one decomposer call and one (shareable) reader call per hop,
        plus the final decomposer call"""
        return sum(1 + self.read(h["subq"], h["rows"]) for h in tn[start:]) + 1

    def regen(self, c, vnew):
        """full recomputation of entry c to version vnew (TTL / provenance): 2m+1 calls, reads shareable"""
        return self.suffix(Ot.trace(c, vnew), 0)


def certify(c, vold, vnew, tab):
    """hop-local certification from the trusted version: re-read every hop whose evidence changed, in order; stop at
    the first hop whose answer changed and recompute only the suffix (exact after any history of earlier decisions)."""
    to, tn = Ot.trace(c, vold), Ot.trace(c, vnew)
    cost = 0
    for j, (ho, hn) in enumerate(zip(to, tn)):
        if ho["subq"] != hn["subq"]:
            return cost + tab.suffix(tn, j)
        if ho["rows"] == hn["rows"]:
            continue
        cost += tab.read(ho["subq"], hn["rows"], verify=True)
        if ho["ans"] != hn["ans"]:
            return cost + tab.suffix(tn, j + 1)
    if len(to) != len(tn) or Ot.ans(c, vold) != Ot.ans(c, vnew):
        return cost + (tab.suffix(tn, len(to)) if len(tn) > len(to) else 1)
    return cost


# ------------------------------------------------------------------ hop scorers with probability calibration
class HopProb:
    """calibrated probability that a cumulative hop edit defeats the cached hop answer.  Raw scores (one or a few
    features) are mapped to probabilities by logistic (Platt) calibration fitted on the hop edits of the calibration
    replay -- the same procedure for every scorer.
    Scorer names:  mtr:<tag> | setcross1:<tag> (cross-encoders of deltar.py);  nli;  margin (trusted margin only);
    uniform;  oracle;  spec:<model>[:<adapter>][:<mode>]  -- the speculator (spec.py), a same-family sibling reader
    measuring the signed margin of the cached tokens; mode zs = its current margin alone; did = margin transport
    (trusted reader margin + the sibling's margin shift K* -> K); cur+ = current sibling margin and trusted margin."""

    def __init__(self, name):
        self.name, self.memo, self.time, self.calls = name, {}, 0.0, 0
        self.model = self.nli = self.spec = None
        self.mode = "zs"
        parts = name.split(":")
        self.alt = None                                   # the obvious alternatives with the same small model
        if parts[0] in ("spec", "judge", "reans"):
            from spec import Speculator
            MODES = ("zs", "did", "cur+", "raw", "ll", "shift")
            adapter = parts[2] if len(parts) > 2 and parts[2] not in MODES else ""
            self.mode = next((p for p in parts[2:] if p in MODES), "zs")
            self.spec = Speculator(parts[1], adapter)
            if parts[0] != "spec":
                self.alt = parts[0]
        elif len(parts) >= 2:
            self.model = load_dr(parts[0], parts[1])
        elif name == "nli":
            self.nli = NLI()
        self.w, self.b = np.ones(1), 0.0
        if name not in ("uniform", "oracle") and not name.endswith(":raw"):   # raw: the model's own probability, no Platt
            self.fit()

    def raw(self, items, texts, book):
        """feature matrix (n, d) for hop items (subq, cached answer, K*, K, mu*)"""
        if self.alt == "judge":                           # same small model asked yes/no: feature = log P(no) - log P(yes)
            sc = -np.array(self.spec.judge(texts, [(q, K2, a) for q, a, K, K2, m in items]))
            return (sc / getattr(self.spec, "scale", 1.0))[:, None] if self.mode == "raw" else sc[:, None]
        if self.alt == "reans":                           # same small model re-answers the hop: feature = answer differs
            gen = self.spec.reanswer(texts, [(q, K2) for q, a, K, K2, m in items])
            return np.array([0.0 if equiv(g, a) else 1.0 for g, (q, a, K, K2, m) in zip(gen, items)])[:, None]
        if self.spec is not None:
            ys = [book.tokens(q, K) or [] for q, a, K, K2, m in items]
            S2 = self.spec.stats(texts, [(q, K2, tuple(y)) for (q, a, K, K2, m), y in zip(items, ys)])
            cur = np.array([s["m"] for s in S2])
            if self.mode == "zs":
                return -cur[:, None]
            if self.mode == "raw":
                return -(cur / getattr(self.spec, "scale", 1.0))[:, None]
            mu = np.log1p(np.maximum(np.array([m for *_, m in items]), 0.0))
            if self.mode == "cur+":
                return -np.stack([cur, mu], 1)
            S1 = self.spec.stats(texts, [(q, K, tuple(y)) for (q, a, K, K2, m), y in zip(items, ys)])
            tru = np.array([s["m"] for s in S1])
            dll = np.array([b["ll"] - a_["ll"] for a_, b in zip(S1, S2)])
            if self.mode == "ll":                                        # likelihood shift of the cached tokens alone
                return -dll[:, None]
            if self.mode == "shift":
                return -(cur - tru)[:, None]
            return -np.stack([cur - tru, mu, cur, dll], 1)              # did: margin shift, trusted reader margin, current margin, likelihood shift
        if self.model is not None:
            exs = [{"subq": q, "ans": a, "K": [(0, x) for x in K], "K2": [(0, x) for x in K2], "mu": m} for q, a, K, K2, m in items]
            return score_examples(self.model, exs, {}, lambda d: texts[d[1]], bs=32)[:, None]
        if self.name == "margin":
            return np.array([-math.log1p(m) for *_, m in items])[:, None]
        if self.name == "nli":                 # pairwise verifier: max contradiction over entering documents
            out = []
            for q, a, K, K2, m in items:
                new = [x for x in K2 if x not in set(K)]
                if not new:
                    out.append(1e4); continue            # evictions/deletions alone cannot be scored pairwise
                p = self.nli([texts[x] for x in new], [f"The answer to the question '{q}' is {a}."] * len(new))
                out.append(float(p.max()))
            return np.array(out)[:, None]
        raise ValueError(self.name)

    def fit(self):
        """logistic calibration on the calibration replay's hop edits (label = the reader's hop answer changes)"""
        from sklearn.linear_model import LogisticRegression
        exs, T = edit_examples(args.wl, "cal")
        MB = MarginBook(args.wl, "cal")
        items = [(e["subq"], e["ans"], tuple(x for _, x in e["K"]), tuple(x for _, x in e["K2"]),
                  MB.mu(e["subq"], [x for _, x in e["K"]])) for e in exs]
        z = np.clip(self.raw(items, T, MB), -50, 50); y = np.array([e["y"] for e in exs])
        lr = LogisticRegression(C=100.0, max_iter=1000).fit(z, y)
        self.w, self.b = lr.coef_[0].astype(float), float(lr.intercept_[0])
        if self.spec is not None:                       # calibration features are per split: forget them
            self.spec.memo.clear(); self.spec.pmemo.clear(); self.spec.time = 0.0; self.spec.calls = 0; self.spec.tokens = 0
        print(self.name, "calib w", np.round(self.w, 4).tolist(), "b", round(self.b, 4), "cal edits", len(y), flush=True)

    def prob(self, items):
        need = [it for it in items if it not in self.memo]
        if need:
            t = time.time()
            z = np.clip(self.raw(need, Ot.texts, MBt), -50, 50)
            for it, x in zip(need, z):
                self.memo[it] = 1.0 / (1.0 + math.exp(-(float(x @ self.w) + self.b)))
            self.time += time.time() - t; self.calls += len(need)
        return [self.memo[it] for it in items]


SHARE = int(os.environ.get("SHARE_LEVEL", 2))     # default engine: shared verification and shared computation


def run(scorer, alpha, rho_min, gamma, seed, share=SHARE, exact=False, log=None, cids=None, stream_=None, rel=None):
    """one closed-loop replay.  cids/stream_: a sub-cache and its request stream (cache-size study);
    rel: list collecting (p_e, stale, rho) of certified requests for the reliability analysis"""
    cids = cids if cids is not None else Ot.cids
    stream_ = stream_ if stream_ is not None else stream
    rng = np.random.default_rng(seed)
    held = {c: 0 for c in cids}
    memo = HopTable(int(share))
    pe_memo = {}
    a_t = alpha
    m = collections.Counter(); W = 0.0; S_hat = 0.0; scored = set()
    ep, episodes = {}, []
    a_traj = []
    for b in range(1, Ot.nB + 1):
        cur = {c: Ot.cur_vidx(c, b) for c in cids}
        held, touched = refresh(held, b)
        for c, v in held.items():                         # stale episodes (cached answer differs from recomputation)
            st_ = v is not None and not equiv(Ot.ans(c, v), Ot.ans(c, cur[c]))
            if st_ and c not in ep:
                ep[c] = [b, 0]
            elif not st_ and c in ep:
                s0, n0 = ep.pop(c); episodes.append((b - s0, n0, False))
        for c in stream_[b - 1]:
            m["q"] += 1
            v = held[c]
            stale = not equiv(Ot.ans(c, v), Ot.ans(c, cur[c]))
            its = [] if exact else unverified_items(c, v)
            if exact:
                unver = isinstance(v, tuple) and any(tuple(r) != tuple(Ot.versions[c][Ot.vi(v)][1]["trace"][h]["rows"]) for h, r, _ in v[2])
            else:
                unver = bool(its)
            if not unver:
                if stale:                                  # cannot happen up to the recorded nondeterminism floor
                    m["stale"] += 1; m["stale_verified_state"] += 1
                a_t = min(1.0, a_t + gamma * alpha)
                if log is not None:
                    log.append((b, "hit", c, None, None, []))
                continue
            scored_now = []
            if exact:
                rho = 1.0
            elif scorer.name == "oracle":
                pe = 1.0 if stale else 0.0                  # a perfect stale probability (lower bound on cost)
                rho = certification_probability(pe, a_t, rho_min)
            else:
                key = (c, v)
                if key not in pe_memo:
                    scored_now = its
                    ps = [1.0] * len(its) if scorer.name == "uniform" else scorer.prob(its)
                    pe_memo[key] = 1.0 - float(np.prod([1.0 - p for p in ps]))
                    m["scored_hops"] += len(its)
                    m["scored_distinct"] += sum(1 for it in its if it not in scored); scored.update(its)
                pe = pe_memo[key]
                rho = certification_probability(pe, a_t, rho_min)
            V = rng.random() < rho
            W += stale * (1 - rho) / rho
            if V:
                if rel is not None and not exact and scorer.name not in ("uniform", "oracle"):
                    rel.append((pe, int(stale), rho))
                cost = certify(c, Ot.vi(v), cur[c], memo)
                m["llm"] += cost; m["cert"] += 1; m["wait"] += 1
                if stale:
                    m["caught"] += 1
                    if c in ep:
                        s0, n0 = ep.pop(c); episodes.append((b - s0, n0, False))
                held[c] = cur[c]
                s_hat = (1 - rho) / rho if stale else 0.0
                if log is not None:
                    log.append((b, "cert", c, Ot.vi(v), cur[c], scored_now))
            else:
                s_hat = 0.0
                if stale:
                    m["stale"] += 1
                    if c in ep:
                        ep[c][1] += 1
                if log is not None:
                    log.append((b, "hit", c, None, None, scored_now))
            S_hat += s_hat
            a_t = min(1.0, a_t + gamma * (alpha - s_hat))
        a_traj.append(a_t)
    for c, (s0, n0) in ep.items():
        episodes.append((Ot.nB + 1 - s0, n0, True))
    N = m["q"]; d = 0.05; bnd = (1 - rho_min) / rho_min
    ctrl = max(0.0, alpha - a_t) / (gamma * N) if gamma > 0 else 0.0
    # Freedman (increments of the stale-serve martingale are <= 1), union over the grid w in {1, 2, 4, ..., >= N b}
    G = 1 + math.ceil(math.log2(max(1.0, N * bnd))); Lg = math.log(G / d)
    w = 2.0 ** math.ceil(math.log2(max(W, 1.0)))
    mart = (math.sqrt(2 * w * Lg) + 2 * Lg) / N
    lens = np.array([e[0] for e in episodes]) if episodes else np.zeros(1)
    serves = np.array([e[1] for e in episodes]) if episodes else np.zeros(1)
    return {"SRR": m["stale"] / N, "SRR_ipw": S_hat / N, "bound": alpha + ctrl + mart, "bound_ctrl": ctrl, "bound_mart": mart,
            "calls_per_update": m["llm"] / n_upd, "cert_per_q": m["cert"] / N, "wait_frac": m["wait"] / N,
            "scored_hops_per_update": m["scored_hops"] / n_upd, "scored_distinct_per_update": m["scored_distinct"] / n_upd,
            "caught": m["caught"],
            "episodes": len(episodes), "ep_len_mean": float(lens.mean()), "ep_len_p95": float(np.percentile(lens, 95)),
            "ep_serves_mean": float(serves.mean()), "ep_serves_p95": float(np.percentile(serves, 95)),
            "ep_serves_max": float(serves.max()), "censored": int(sum(e[2] for e in episodes)),
            "a_mean": float(np.mean(a_traj)), "a_final": a_t, "N": N, "W": W,
            "stale_verified_state": m["stale_verified_state"]}


def agg(runs):
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out.update({k + "_std": float(np.std([r[k] for r in runs])) for k in ("SRR", "calls_per_update", "bound", "ep_serves_mean") if k in runs[0]})
    out["SRR_max_seed"] = float(max(r["SRR"] for r in runs))
    return out


# ------------------------------------------------------------------ published baselines, same stream / oracle / metrics
# Every baseline that re-verifies uses the same hop-local, state-shared certification engine as DEFT (certify), so the
# comparison isolates the serve-or-verify decision; regeneration-based baselines (TTL, provenance) pay a full
# recomputation of the answer pipeline, as in their native systems.
class VCacheEntry:
    """per-entry metadata of vCache (observations of (similarity, correct))"""
    def __init__(self):
        self.observations = []; self.gamma = None; self.t_hat = None; self.t_prime = None; self.var_t = None


class Verifier:
    """verify-on-hit with a published grounding checker: P(cached hop answer supported by the current evidence)"""
    def __init__(self, name):
        import baselines as BL
        self.name, self.memo, self.time, self.calls = name, {}, 0.0, 0
        if name == "minicheck":
            self.m = BL.MiniCheck()
        elif name == "hhem":
            self.m = BL.HHEM()
        else:
            self.m = None; self.nli = NLI()

    def probs(self, items):
        need = [it for it in items if it not in self.memo]
        if need:
            t = time.time()
            if self.m is not None:
                import baselines as BL
                ps = self.m.support([([Ot.texts[x] for x in K2], BL.CLAIM.format(q=q, a=a)) for q, a, K, K2, m in need])
            else:   # DeBERTa-v3-large NLI, entailment of the claim by the evidence set
                ps = []
                for q, a, K, K2, m in need:
                    pr = self.nli.probs([" ".join(Ot.texts[x] for x in K2)], [f"The answer to the question '{q}' is {a}."])
                    ps.append(float(pr[0, self.nli.ei]))
            for it, p in zip(need, ps):
                self.memo[it] = p
            self.time += time.time() - t; self.calls += len(need)
        return [self.memo[it] for it in items]


class PromptSim:
    """vCache similarity: cosine of the bge embeddings of the cached and the current hop prompts (sub-question + evidence)"""
    def __init__(self):
        self.enc = Encoder(max_len=512); self.memo = {}

    def emb(self, q, K):
        k = (q, K)
        if k not in self.memo:
            self.memo[k] = self.enc.encode([q + " " + " ".join(Ot.texts[x] for x in K)])[0].float()
        return self.memo[k]

    def sim(self, items):
        return min(float(self.emb(q, K) @ self.emb(q, K2)) for q, a, K, K2, m in items)


def fit_freshcache(O):
    """FreshCache (arXiv:2607.04281): P(stale | age) from the calibration stream.
    Exponential hazard, and the paper's MLP refinement over (age, hop count). The closed loop uses the MLP."""
    from sklearn.neural_network import MLPClassifier
    X, y = [], []
    for c in O.cids:
        last = 0
        for b in range(1, O.nB + 1):
            ch = bool(O.logs[b - 1]["changed"].get(c))
            X.append((b - last, len(O.trace(c, O.cur_vidx(c, b - 1)))))
            y.append(int(ch))
            if ch:
                last = b
    X, y = np.array(X, dtype=float), np.array(y)
    mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=500, random_state=0).fit(X / [O.nB, 4.0], y)
    cache = {}

    def p_mlp(a, m):
        if (a, m) not in cache:
            hz = mlp.predict_proba(np.array([[i, m] for i in range(1, a + 1)], dtype=float) / [O.nB, 4.0])[:, 1] if a > 0 else np.zeros(0)
            cache[(a, m)] = 1 - float(np.prod(1 - hz))
        return cache[(a, m)]
    return p_mlp


def grounded_ok(c, v, b):
    """GroundedCache (arXiv:2605.27494) at the published gates, applied to every hop of the cached trace.
    G2 Jaccard of evidence ids >= 0.5, G3 shared documents keep their version, G4 lexical support >= 0.6.
    The same question passes G1. Thresholds are the paper's, not tuned on the test stream."""
    key = (c, Ot.vi(v), b)
    hit = GROUNDED.get(key)
    if hit is not None:
        return hit
    ok = True
    for hop in Ot.versions[c][Ot.vi(v)][1]["trace"]:
        rows, _ = topk_rows(hop["subq"], b)
        docs_new = {Ot.doc_of_row[r] for r in rows}
        docs_old = set(hop["docs"])
        jac = len(docs_new & docs_old) / max(1, len(docs_new | docs_old))
        same_ver = all(r in hop["rows"] for r in rows if Ot.doc_of_row[r] in docs_old)
        at = content_tokens(hop["ans"])
        ev = set().union(*[content_tokens(Ot.texts[r]) for r in rows]) if rows else set()
        sup = len(at & ev) / max(1, len(at)) if at else 1.0
        if jac < 0.5 or not same_ver or sup < 0.6:
            ok = False
            break
    GROUNDED[key] = ok
    return ok


GROUNDED = {}


def run_base(kind, knob, seed, aux=None, cids=None, stream_=None):
    import random as _r
    cids = cids if cids is not None else Ot.cids
    stream_ = stream_ if stream_ is not None else stream
    rng = np.random.default_rng(seed); _r.seed(seed)
    held = {c: 0 for c in cids}; last = {c: 0 for c in cids}
    dirty = set()
    memo = HopTable(SHARE); m = collections.Counter(); md = collections.defaultdict(VCacheEntry)
    alg = None
    if kind.startswith("vcache"):
        from vcache_alg import _Algorithm, _Action
        alg = _Algorithm(delta=knob)
    for b in range(1, Ot.nB + 1):
        cur = {c: Ot.cur_vidx(c, b) for c in cids}
        held, touched = refresh(held, b)
        if kind == "fincache":
            rem = set(Ot.logs[b - 1]["rem"])
            if rem:
                for c in cids:
                    tr = Ot.versions[c][Ot.vi(held[c])][1]["trace"]
                    if any(x in rem for hop in tr for x in hop["rows"]):
                        dirty.add(c)
        for c in stream_[b - 1]:
            m["q"] += 1
            v = held[c]
            stale = not equiv(Ot.ans(c, v), Ot.ans(c, cur[c]))
            its = unverified_items(c, v)
            act = "serve"
            if kind == "ttl":
                if b - last[c] >= knob:
                    act = "regen"
            elif kind == "provenance":
                if its:
                    act = "regen"
            elif kind == "fincache":
                if c in dirty:
                    act = "regen"
            elif kind == "grounded":
                if not grounded_ok(c, v, b):
                    act = "regen"
            elif kind == "fresh":
                tr = Ot.versions[c][Ot.vi(v)][1]["trace"]
                if aux(b - last[c], len(tr)) > knob:
                    act = "regen"
            elif kind.startswith("vcache"):
                if its:                                    # vcache_g: one global metadata object (vCache's global variant)
                    s = aux.sim(its); mo = md["*"] if kind == "vcache_g" else md[c]
                    if alg.select_action(s, mo) == _Action.EXPLORE:
                        act = "cert"; alg.add_observation_to_metadata(s, not stale, mo)
            elif kind == "verify":
                if its:
                    m["scored_hops"] += len(its)
                    if min(aux.probs(its)) < knob:
                        act = "cert"
            if act == "regen":
                m["llm"] += memo.regen(c, cur[c]); m["wait"] += 1
                held[c] = cur[c]; last[c] = b
                dirty.discard(c)
            elif act == "cert":
                m["llm"] += certify(c, Ot.vi(v), cur[c], memo); m["wait"] += 1
                held[c] = cur[c]; last[c] = b
            elif stale:
                m["stale"] += 1
    N = m["q"]
    return {"SRR": m["stale"] / N, "calls_per_update": m["llm"] / n_upd, "wait_frac": m["wait"] / N,
            "scored_hops_per_update": m["scored_hops"] / n_upd, "N": N}


def base_sweeps(names, on_done=None):
    out = {}
    for nm in names:
        res = {}
        if nm == "gptcache":
            res["-"] = run_base("serve", 0, 0)
        elif nm == "ttl":
            for T in (1, 2, 4, 8, 16, 32):
                res[str(T)] = run_base("ttl", T, 0)
        elif nm == "provenance":
            res["-"] = run_base("provenance", 0, 0)
        elif nm == "fincache":
            res["-"] = run_base("fincache", 0, 0)
        elif nm == "grounded":
            res["-"] = run_base("grounded", 0, 0)
        elif nm == "freshcache":
            Oc = Oracle(args.wl, "cal")
            fc = fit_freshcache(Oc)
            del Oc
            torch.cuda.empty_cache()
            res["0.01"] = run_base("fresh", 0.01, 0, fc)
        elif nm in ("vcache", "vcache_g"):
            ps = PromptSim()
            for d in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
                res[str(d)] = agg([run_base(nm, d, s, ps) for s in range(args.seeds)])
            del ps
        else:
            vf = Verifier(nm)
            for tau in (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999):
                res[str(tau)] = run_base("verify", tau, 0, vf)
            res["score_ms_per_hop"] = 1000 * vf.time / max(1, vf.calls)
            del vf; torch.cuda.empty_cache()
        out[nm] = res
        print("base", nm, {k: (round(v["SRR"], 4), round(v["calls_per_update"], 2)) for k, v in res.items() if isinstance(v, dict)}, flush=True)
        if on_done is not None:
            on_done(nm, res)                       # persist each baseline as soon as it is finished
    return out


R = {"meta": {"wl": args.wl, "n_updates": n_upd, "N": sum(len(s) for s in stream)}}
dst = f"{OUT}/{args.out}.json"
if os.path.exists(dst):
    R.update(jload(dst))
R["meta"]["share_level"] = SHARE
R["exact"] = run(None, 0, 1, 0, 0, share=0, exact=True)
R["exact+share"] = run(None, 0, 1, 0, 0, share=1, exact=True)            # shared verification only
R["exact+share2"] = run(None, 0, 1, 0, 0, share=2, exact=True)           # shared verification and computation
for k in ("exact", "exact+share", "exact+share2"):
    print(k, {kk: round(v, 4) for kk, v in R[k].items() if isinstance(v, float)}, flush=True)
jdump(R, dst)
if args.baselines:
    def _save(nm, res):
        R["base:" + nm] = res
        jdump(R, dst)
    base_sweeps(args.baselines.split(","), on_done=_save)
def reliability_bins(rel, edges=(0, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0001)):
    """IPW-weighted calibration of p_e on certified requests: each certified request stands for 1/rho requests"""
    out = []
    rel = np.array(rel) if rel else np.zeros((0, 3))
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (rel[:, 0] >= lo) & (rel[:, 0] < hi) if len(rel) else np.zeros(0, bool)
        if sel.sum() == 0:
            continue
        w = 1.0 / rel[sel, 2]
        out.append({"lo": lo, "hi": min(hi, 1.0), "n": int(sel.sum()), "mean_p": float((w * rel[sel, 0]).sum() / w.sum()),
                    "stale_rate": float((w * rel[sel, 1]).sum() / w.sum())})
    return out


def size_study(sc, sizes, nseeds):
    """calls per update as a function of the number of cached questions, same update stream: per-entry policies grow
    linearly with the cache, policies on the shared hop-state table grow sublinearly"""
    out = {}
    for n in sizes:
        rows = collections.defaultdict(list)
        for s in range(nseeds):
            rng = np.random.default_rng(100 + s)
            sub = list(rng.choice(Ot.cids, n, replace=False)) if n < len(Ot.cids) else list(Ot.cids)
            st = zipf_stream(sub, Ot.nB, max(1, int(0.5 * len(sub))), seed=s)
            rows["exact"].append(run(None, 0, 1, 0, 0, share=0, exact=True, cids=sub, stream_=st)["calls_per_update"])
            rows["exact+share"].append(run(None, 0, 1, 0, 0, share=1, exact=True, cids=sub, stream_=st)["calls_per_update"])
            rows["exact+share2"].append(run(None, 0, 1, 0, 0, share=2, exact=True, cids=sub, stream_=st)["calls_per_update"])
            rows["provenance"].append(run_base("provenance", 0, 0, cids=sub, stream_=st)["calls_per_update"])
            rows["uniform"].append(float(np.mean([run(HopProb("uniform"), 0.01, 0.05, 0.01, k, cids=sub, stream_=st)["calls_per_update"] for k in range(3)])))
            if sc is not None:
                rr = [run(sc, 0.01, 0.05, 0.01, k, cids=sub, stream_=st) for k in range(3)]
                rows["deft"].append(float(np.mean([r["calls_per_update"] for r in rr])))
                rows["deft_srr"].append(float(np.mean([r["SRR"] for r in rr])))
                rows["deft_scored"].append(float(np.mean([r["scored_distinct_per_update"] for r in rr])))
        out[str(n)] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in rows.items()}
        print("size", n, {k: round(v["mean"], 3) for k, v in out[str(n)].items()}, flush=True)
    return out


first = True
for name in [x for x in args.scorers.split(",") if x]:
    try:
        sc = HopProb(name)
    except Exception as e:
        print("skip", name, repr(e), flush=True); continue
    res = R.get(name, {})
    if args.cache_sizes:
        R["sizes:" + name] = size_study(sc, [int(x) for x in args.cache_sizes.split(",")] + [len(Ot.cids)], args.size_seeds)
        jdump(R, dst)
    if args.reliability:
        rel = []
        for s in range(args.seeds):
            run(sc, 0.01, 0.05, 0.01, s, rel=rel)
        res["reliability"] = reliability_bins(rel)
        print(name, "reliability", [(round(b["mean_p"], 3), round(b["stale_rate"], 3), b["n"]) for b in res["reliability"]], flush=True)
    for a in [float(x) for x in args.alphas.split(",")]:
        for rm in [float(x) for x in args.rhos.split(",")]:
            for g in [float(x) for x in args.gammas.split(",")]:
                key = f"{a}|{rm}|{g}"
                res[key] = agg([run(sc, a, rm, g, s) for s in range(args.seeds)])
                print(name, key, {k: round(v, 4) for k, v in res[key].items() if k in ("SRR", "bound", "calls_per_update", "wait_frac", "ep_serves_mean", "SRR_max_seed", "scored_distinct_per_update")}, flush=True)
                if first and args.noshare and a == 0.01 and rm == 0.05:
                    res[key + "|noshare"] = agg([run(sc, a, rm, g, s, share=0) for s in range(args.seeds)])
                    res[key + "|share1"] = agg([run(sc, a, rm, g, s, share=1) for s in range(args.seeds)])
    res["score_ms_per_hop"] = 1000 * sc.time / max(1, sc.calls)
    res["scored_hops"] = sc.calls
    res["calib"] = {"w": sc.w.tolist(), "b": sc.b}
    if sc.spec is not None:
        res["spec_ms_per_hop"] = 1000 * sc.spec.time / max(1, sc.spec.calls)
        res["spec_tokens_per_hop"] = sc.spec.tokens / max(1, sc.spec.calls)
        res["spec_forwards"] = sc.spec.calls
    R[name] = res
    jdump(R, dst)
    if args.dump and name == args.dump:
        import pickle
        lg_ex = []; run(None, 0, 1, 0, 0, share=SHARE, exact=True, log=lg_ex)
        lg_x = []; run(sc, 0.01, 0.05, float(args.gammas.split(",")[0]), 0, log=lg_x)
        pickle.dump({"exact": lg_ex, "deft": lg_x}, open(f"{OUT}/reqlog.pkl", "wb"))
        pickle.dump({"exact": lg_ex, "deft": lg_x}, open(f"{OUT}/reqlog_{name.replace(':', '_')}.pkl", "wb"))
    first = False
    del sc; torch.cuda.empty_cache()
print("done", flush=True)
