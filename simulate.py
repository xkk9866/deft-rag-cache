"""Replay an update/query stream under every cache-maintenance policy and measure consistency and cost.

Ground truth comes from oracle.pkl (exact, recompute-consistent answers after every batch).
All policies share the same query stream and the same LLM recomputation results (a recomputation at batch b
always returns the oracle version valid at b), so policies differ only in *which* entries they invalidate.
"""
import argparse, random, re, time, collections
import numpy as np
from common import *

STOP = set("the of and a an in on at to for from by with is was are were be been as that this which who what when where how "
           "whose whom it its into than then also not no or but".split())


def content_tokens(s):
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 3 and t not in STOP}


ENT = re.compile(r"\b([A-Z][\w\-\.'’]*(?:\s+(?:of|the|de|del|van|von|da|and|[A-Z][\w\-\.'’]*))*)")
GENERIC = {"The", "In", "As", "On", "At", "It", "He", "She", "They", "This", "A", "An", "Before", "After", "Who", "What",
           "Which", "When", "Where", "How", "Name", "Answer", "Question", "News", "Today", "Current", "Currently"}


def entities(s):
    out = set()
    for m in ENT.findall(s):
        m = m.strip()
        if m in GENERIC or len(m) < 3 or m.isdigit():
            continue
        out.add(m.lower())
    return out


class Oracle:
    def __init__(self, wl, split):
        self.R = R = pload(f"{RUNS}/{wl}/{split}/oracle.pkl")
        self.wl, self.split = wl, split
        self.cases = R["cases"]; self.cids = [c["cid"] for c in self.cases]
        self.case = {c["cid"]: c for c in self.cases}
        self.logs = R["logs"]; self.nB = len(self.logs)
        self.E = torch.tensor(R["E"]).float().cuda()
        self.texts = R["texts"]; self.doc_of_row = R["doc_of_row"]
        self.qemb = {q: torch.tensor(v).float().cuda() for q, v in R["qemb"].items()}
        self.versions = R["versions"]           # cid -> [(b, res)]
        self.k = R["k"]
        # alive mask after each batch
        alive = torch.zeros(len(self.texts), dtype=torch.bool, device="cuda"); alive[: R["init_rows"]] = True
        self.alive = [alive.clone()]
        for lg in self.logs:
            for row, _, _ in lg["ins"]:
                alive[row] = True
            for row in lg["rem"]:
                alive[row] = False
            self.alive.append(alive.clone())
        self.ins_rows = [[row for row, _, _ in lg["ins"]] for lg in self.logs]
        self.ins_kind = {row: kind for lg in self.logs for row, kind, _ in lg["ins"]}
        self.ev_kinds = [collections.Counter(k for _, k, _ in lg["events"]) for lg in self.logs]
        # re-derive defeat flags under answer equivalence
        for b, lg in enumerate(self.logs, start=1):
            lg["changed"] = {cid: not equiv(self.ans(cid, self.cur_vidx(cid, b - 1)), self.ans(cid, self.cur_vidx(cid, b)))
                             for cid in lg["reach"]}

    def cur_vidx(self, cid, b):
        """index of oracle version valid after batch b"""
        vs = self.versions[cid]; j = 0
        for i, (vb, _) in enumerate(vs):
            if vb <= b:
                j = i
        return j

    # a cache state is either an oracle version index, or a refreshed state ("R", vidx, overrides) in which some hops'
    # evidence sets were replaced by their exact post-edit sets K' while the cached answer was retained
    @staticmethod
    def vi(v):
        return v[1] if isinstance(v, tuple) else v

    def ans(self, cid, vidx):
        return self.versions[cid][self.vi(vidx)][1]["answer"]

    def trace(self, cid, vidx):
        if not isinstance(vidx, tuple):
            return self.versions[cid][vidx][1]["trace"]
        key = (cid, vidx)
        if not hasattr(self, "_tcache"):
            self._tcache = {}
        if key not in self._tcache:
            tr = [dict(h) for h in self.versions[cid][vidx[1]][1]["trace"]]
            for h, rows, scores in vidx[2]:
                tr[h]["rows"] = list(rows); tr[h]["scores"] = list(scores); tr[h]["tau"] = scores[-1]
                tr[h]["docs"] = [self.doc_of_row[x] for x in rows]
            self._tcache[key] = tr
        return self._tcache[key]

    def calls(self, cid, vidx):
        return 2 * len(self.trace(cid, vidx)) + 1


class Scorer:
    """Pair scores s(hop key of (cid, vidx, h), inserted row); higher = more likely defeated."""

    def __init__(self, O, kind, model=None, nli=None):
        self.O, self.kind, self.model, self.nli = O, kind, model, nli
        self.kcache, self.pcache, self.pair = {}, {}, {}
        self.time = 0.0

    def keys(self, items):
        miss = [it for it in items if it not in self.kcache]
        if miss:
            subqs = [self.O.trace(c, v)[h]["subq"] for c, v, h in miss]
            ans = [self.O.trace(c, v)[h]["ans"] for c, v, h in miss]
            ev = [" ".join(self.O.texts[x] for x in self.O.trace(c, v)[h]["rows"]) for c, v, h in miss]
            K = self.model.key_vectors(subqs, ans, ev if getattr(self.model, "frag", False) else None)
            for it, kv in zip(miss, K):
                self.kcache[it] = kv
        return torch.stack([self.kcache[it] for it in items])

    def probes(self, rows):
        miss = [r for r in rows if r not in self.pcache]
        if miss:
            P = self.model.probe_vectors([self.O.texts[r] for r in miss])
            for r, pv in zip(miss, P):
                self.pcache[r] = pv
        return torch.stack([self.pcache[r] for r in rows])

    def score_pairs(self, pairs):
        """pairs: list of ((cid, vidx, h), row) -> np.array scores"""
        t = time.time()
        if not pairs:
            return np.zeros(0)
        if self.kind in ("svde", "svde_noaux", "bienc", "svdef", "svdef_noaux"):
            items = sorted({p[0] for p in pairs}); rows = sorted({p[1] for p in pairs})
            K = self.keys(items); P = self.probes(rows)
            S = (K @ P.T).cpu().numpy()
            ii = {it: i for i, it in enumerate(items)}; rr = {r: i for i, r in enumerate(rows)}
            out = np.array([S[ii[a], rr[b]] for a, b in pairs])
        elif self.kind in ("cross", "nli"):
            need = [p for p in pairs if p not in self.pair]
            if need:
                subqs = [self.O.trace(c, v)[h]["subq"] for (c, v, h), r in need]
                ans = [self.O.trace(c, v)[h]["ans"] for (c, v, h), r in need]
                docs = [self.O.texts[r] for _, r in need]
                sc = []
                with torch.no_grad():
                    for i in range(0, len(need), 64):
                        if self.kind == "cross":
                            sc.append(self.model.cross_logits(subqs[i:i+64], ans[i:i+64], docs[i:i+64]).float().cpu())
                        else:
                            sc.append(self.nli(docs[i:i+64], [f"The answer to the question '{q}' is {a}." for q, a in zip(subqs[i:i+64], ans[i:i+64])]))
                sc = torch.cat(sc).numpy()
                for p, s in zip(need, sc):
                    self.pair[p] = float(s)
            out = np.array([self.pair[p] for p in pairs])
        elif self.kind == "margin":
            out = np.array([float(self.O.qemb[self.O.trace(c, v)[h]["subq"]] @ self.O.E[r] - self.O.trace(c, v)[h]["tau"]) for (c, v, h), r in pairs])
        else:
            raise ValueError(self.kind)
        self.time += time.time() - t
        return out


class NLI:
    def __init__(self, path=f"{ROOT}/models/deberta-v3-large-nli"):
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).cuda().eval()
        lab = {v.lower(): int(k) for k, v in self.m.config.id2label.items()}
        self.ci = lab.get("contradiction", 2); self.ei = lab.get("entailment", 0)

    @torch.no_grad()
    def probs(self, prem, hyp, max_length=512):
        b = self.tok(prem, hyp, padding=True, truncation="only_first", max_length=max_length, return_tensors="pt").to("cuda")
        return torch.softmax(self.m(**b).logits.float(), -1).cpu()

    @torch.no_grad()
    def __call__(self, prem, hyp):
        b = self.tok(prem, hyp, padding=True, truncation=True, max_length=320, return_tensors="pt").to("cuda")
        return torch.softmax(self.m(**b).logits.float(), -1)[:, self.ci].cpu()


# ------------------------------------------------------------------ reachability (Thm. 1)
def reach_pairs(O, held, b):
    """all (item=(cid,vidx,h), row) with inserted row at batch b reaching the held trace; plus provenance hits.
    Reachability of a fixed (entry version, batch) is deterministic and memoised across replays."""
    if not hasattr(O, "_rmemo"):
        O._rmemo, O._pmemo, O._hkey = {}, {}, {}
    rows = O.ins_rows[b - 1]
    need = [(cid, v) for cid, v in held.items() if v is not None and (cid, v, b) not in O._rmemo]
    if need:
        owners, K = [], []
        for cid, v in need:
            if (cid, v) not in O._hkey:
                tr = O.trace(cid, v)
                O._hkey[(cid, v)] = torch.stack([torch.cat([O.qemb[hop["subq"]], torch.tensor([-hop["tau"]], device="cuda")]) for hop in tr]) if tr else None
            if O._hkey[(cid, v)] is not None:
                K.append(O._hkey[(cid, v)]); owners += [(cid, v, h) for h in range(len(O._hkey[(cid, v)]))]
        for cid, v in need:
            O._rmemo[(cid, v, b)] = []
        if rows and K:
            K = torch.cat(K)
            P = torch.cat([O.E[rows], torch.ones(len(rows), 1, device="cuda")], 1)
            M = (K @ P.T) >= -1e-4
            for oi, ri in M.nonzero().tolist():
                c, v, h = owners[oi]
                O._rmemo[(c, v, b)].append(((c, v, h), rows[ri]))
        rem = set(O.logs[b - 1]["rem"])
        for cid, v in need:
            O._pmemo[(cid, v, b)] = bool(rem) and any(x in rem for hop in O.trace(cid, v) for x in hop["rows"])
    pairs, prov = [], set()
    for cid, v in held.items():
        if v is None:
            continue
        pairs += O._rmemo[(cid, v, b)]
        if O._pmemo[(cid, v, b)]:
            prov.add(cid)
    return pairs, prov


def all_pairs(O, held, b):
    rows = O.ins_rows[b - 1]
    return [((cid, v, h), r) for cid, v in held.items() if v is not None for h in range(len(O.trace(cid, v))) for r in rows]


# ------------------------------------------------------------------ mechanism analysis
def kind_breakdown(O, scorer, lam, use_reach=True):
    """one-step, pair level: for each kind of inserted document, how many reachable (hop, doc) pairs exist,
    how many belong to an entry that is actually defeated in that batch, and how many the scorer flags."""
    out = collections.defaultdict(lambda: collections.Counter())
    for b in range(1, O.nB + 1):
        held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
        pairs, _ = reach_pairs(O, held, b)
        if not pairs:
            continue
        s = scorer.score_pairs(pairs) if scorer is not None else np.full(len(pairs), np.inf)
        chg = O.logs[b - 1]["changed"]
        for ((c, v, h), r), x in zip(pairs, s):
            kd = O.ins_kind.get(r, "?")
            out[kd]["pairs"] += 1
            out[kd]["defeated_entry"] += int(bool(chg.get(c, False)))
            out[kd]["flagged"] += int(x >= lam)
    return {k: dict(v) for k, v in out.items()}


# ------------------------------------------------------------------ calibration (conformal miss control)
def defeat_scores(O, scorer, use_reach=True, qsim=None, entity=False):
    """one-step scores of every newly defeated entry: max pair score over the batch (inf if provenance-flagged)."""
    out = []
    for b in range(1, O.nB + 1):
        held = {cid: O.cur_vidx(cid, b - 1) for cid in O.cids}
        defeated = [cid for cid, c in O.logs[b - 1]["changed"].items() if c]
        if not defeated:
            continue
        sub = {cid: held[cid] for cid in defeated}
        s = entry_scores(O, sub, b, scorer, use_reach, qsim, entity)
        out += [s[cid] for cid in defeated]
    return np.array(out)


def answer_rows(O, c, v, h):
    """the evidence set the cached hop answer was derived from (its provenance), as opposed to the refreshed current
    set used for reachability: the differential reader always judges the cumulative edit since the answer was produced,
    so a defeat missed once is re-examined at the entry's next touch and accumulated evictions are visible."""
    if os.environ.get("DR_BEFORE") == "current":
        return O.trace(c, v)[h]["rows"]
    return O.versions[c][O.vi(v)][1]["trace"][h]["rows"]


class SetScorer:
    """Delta-Reader scorer: every hop touched by batch b (reached by an insertion or citing a removed row) gets its
    exact post-update evidence set K' (top-k over the live corpus); the entry score is the max defeat logit over its
    touched hops.  Insertions, deletions and modifications are handled uniformly as evidence-set edits."""
    set_level = True

    def __init__(self, O, model):
        from deltar import score_examples
        self.O, self.model, self.score_examples = O, model, score_examples
        self.cache, self.memo = {}, {}
        self.time = 0.0
        self.texts_of = lambda d: O.texts[d[1]]

    def margins(self):
        """trusted margins of the cached hop answers (only the margin-transport reader reads them)"""
        if not str(getattr(self.model, "variant", "")).startswith("mtr"):
            return None
        if not hasattr(self, "_mb"):
            from mcert import MarginBook
            self._mb = MarginBook(self.O.wl, self.O.split)
        return self._mb

    def touched(self, held, b):
        O = self.O
        pairs, _ = reach_pairs(O, held, b)
        hops = {it for it, _ in pairs}
        rem = set(O.logs[b - 1]["rem"])
        if rem:
            for cid, v in held.items():
                if v is None:
                    continue
                for h, hop in enumerate(O.trace(cid, v)):
                    if any(x in rem for x in hop["rows"]):
                        hops.add((cid, v, h))
        return sorted(hops)

    def entry_scores(self, held, b):
        O = self.O
        t = time.time()
        sc = {cid: -np.inf for cid, v in held.items() if v is not None}
        hops = self.touched(held, b)
        need = [it for it in hops if (it, b) not in self.memo]
        if need:
            Q = torch.stack([O.qemb[O.trace(c, v)[h]["subq"]] for c, v, h in need])
            s = Q @ O.E.T
            s[:, ~O.alive[b]] = -1e4
            top = torch.topk(s, O.k, dim=1).indices.tolist()
            exs = []
            mb = self.margins()
            for (c, v, h), k2 in zip(need, top):
                hop = O.trace(c, v)[h]
                K = answer_rows(O, c, v, h)
                exs.append({"subq": hop["subq"], "ans": hop["ans"], "K": [(0, x) for x in K], "K2": [(0, x) for x in k2],
                            "mu": mb.mu(hop["subq"], K) if mb else 0.0})
            z = self.score_examples(self.model, exs, self.cache, self.texts_of)
            for it, x in zip(need, z):
                self.memo[(it, b)] = float(x)
        for c, v, h in hops:
            sc[c] = max(sc[c], self.memo[((c, v, h), b)])
        self.time += time.time() - t
        return sc


class SetNLIScorer(SetScorer):
    """zero-shot set-level NLI on the same exact evidence edits: premise = the evidence set, hypothesis = the cached
    hop answer; score = entailment lost by the edit + contradiction gained,  P_e(K) - P_e(K') + P_c(K')."""

    def __init__(self, O, nli):
        self.O, self.nli = O, nli
        self.memo, self.time = {}, 0.0
        self.pmemo = {}

    def entry_scores(self, held, b):
        O = self.O
        t = time.time()
        sc = {cid: -np.inf for cid, v in held.items() if v is not None}
        hops = self.touched(held, b)
        need = [it for it in hops if (it, b) not in self.memo]
        if need:
            Q = torch.stack([O.qemb[O.trace(c, v)[h]["subq"]] for c, v, h in need])
            s = Q @ O.E.T
            s[:, ~O.alive[b]] = -1e4
            top = torch.topk(s, O.k, dim=1).indices.tolist()
            jobs = {}
            for (c, v, h), k2 in zip(need, top):
                hop = O.trace(c, v)[h]
                hyp = f"The answer to the question '{hop['subq']}' is {hop['ans']}."
                for rows in (tuple(answer_rows(O, c, v, h)), tuple(k2)):
                    if (rows, hyp) not in self.pmemo:
                        jobs[(rows, hyp)] = None
            keys = list(jobs)
            for i in range(0, len(keys), 32):
                ch = keys[i:i + 32]
                P = self.nli.probs([" ".join(O.texts[x] for x in r) for r, _ in ch], [h for _, h in ch])
                for kk, p in zip(ch, P):
                    self.pmemo[kk] = (float(p[self.nli.ei]), float(p[self.nli.ci]))
            for (c, v, h), k2 in zip(need, top):
                hop = O.trace(c, v)[h]
                hyp = f"The answer to the question '{hop['subq']}' is {hop['ans']}."
                e1, _ = self.pmemo[(tuple(answer_rows(O, c, v, h)), hyp)]; e2, c2 = self.pmemo[(tuple(k2), hyp)]
                self.memo[((c, v, h), b)] = e1 - e2 + c2
        for c, v, h in hops:
            sc[c] = max(sc[c], self.memo[((c, v, h), b)])
        self.time += time.time() - t
        return sc


def entry_scores(O, held, b, scorer, use_reach=True, qsim=None, entity=False):
    if getattr(scorer, "set_level", False):
        return scorer.entry_scores(held, b)
    pairs, prov = reach_pairs(O, held, b)
    sc = {cid: -np.inf for cid, v in held.items() if v is not None}
    if qsim is not None:                                    # question-level reverse similarity
        rows = O.ins_rows[b - 1]
        if rows:
            cids = [c for c in sc]
            Q = torch.stack([qsim[c] for c in cids]); S = (Q @ O.E[rows].T).max(1).values.cpu().numpy()
            for c, s in zip(cids, S):
                sc[c] = float(s)
    elif entity:
        rows = O.ins_rows[b - 1]
        dents = [entities(O.texts[r]) for r in rows]
        for c in sc:
            v = held[c]
            ents = entities(O.case[c]["q"]) | {e for hop in O.trace(c, v) for e in entities(hop["subq"]) | {hop["ans"].lower()}}
            sc[c] = 1.0 if any(ents & d for d in dents) else -np.inf
    elif scorer is None:                                    # pure trace-exact
        for (c, v, h), r in pairs:
            sc[c] = np.inf
    else:
        P = pairs if use_reach else all_pairs(O, held, b)
        s = scorer.score_pairs(P)
        for ((c, v, h), r), x in zip(P, s):
            sc[c] = max(sc[c], float(x))
    for c in prov:
        sc[c] = np.inf
    return sc


def defeat_scores_batched(O, scorer, use_reach=True, qsim=None):
    """per batch: scores of the entries the batch defeats (only batches with at least one defeat)."""
    out = []
    for b in range(1, O.nB + 1):
        defeated = [cid for cid, c in O.logs[b - 1]["changed"].items() if c]
        if not defeated:
            continue
        sub = {cid: O.cur_vidx(cid, b - 1) for cid in defeated}
        s = entry_scores(O, sub, b, scorer, use_reach, qsim)
        out.append(np.array([s[c] for c in defeated]))
    return out


def crc_threshold(batches, alpha):
    """Conformal risk control with batches as exchangeable units; loss = missed fraction of the batch's
    defeated entries (bounded by 1). Largest lambda with n/(n+1) R_n(lambda) + 1/(n+1) <= alpha."""
    n = len(batches)
    if n == 0 or 1.0 / (n + 1) > alpha:
        return -np.inf
    cand = np.unique(np.concatenate([x[np.isfinite(x)] for x in batches] + [np.array([-np.inf])]))
    best = -np.inf
    for lam in cand:
        risk = np.mean([np.mean(x < lam) for x in batches])
        if n / (n + 1) * risk + 1.0 / (n + 1) <= alpha:
            best = lam
        else:
            break
    return float(best)


def conformal_threshold(scores, alpha):
    """largest lambda with P(miss) <= alpha for an exchangeable new defeated entry (CRC with 0-1 miss loss)."""
    n = len(scores)
    kk = int(np.floor(alpha * (n + 1)))
    if kk < 1:
        return -np.inf
    return float(np.sort(scores)[kk - 1])


# ------------------------------------------------------------------ policy replay
def zipf_stream(cids, nB, per_batch, s=1.1, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(cids))
    w = 1.0 / np.arange(1, len(cids) + 1) ** s; w /= w.sum()
    return [[cids[perm[i]] for i in rng.choice(len(cids), per_batch, p=w)] for _ in range(nB)]


def certify_cost(O, c, vold, vnew):
    """hop-local certification of a suspect entry (Thm. 1): re-read only hops whose evidence set changed, in order;
    if every re-read hop answer is unchanged the entry is fresh, otherwise recompute only the suffix."""
    to, tn = O.trace(c, vold), O.trace(c, vnew)
    cost = 0
    for j, (ho, hn) in enumerate(zip(to, tn)):
        if ho["subq"] != hn["subq"]:
            return cost + O.calls(c, vnew) - 2 * j, False
        if ho["rows"] == hn["rows"]:
            continue
        cost += 1
        if ho["ans"] != hn["ans"]:
            return cost + O.calls(c, vnew) - 2 * j - 2, False
    if len(to) != len(tn) or O.ans(c, vold) != O.ans(c, vnew):
        return cost + max(1, O.calls(c, vnew) - 2 * min(len(to), len(tn))), False
    return cost, True


def refresh_states(O, held, b, retained):
    """State maintenance invariant: every retained entry whose evidence was touched by batch b gets its touched hops'
    evidence sets, scores and thresholds replaced by the exact post-edit sets K' (Thm. 1), so that reachability and
    the differential reader of later batches are evaluated against the evidence the cache currently relies on."""
    sub = {c: held[c] for c in retained}
    pairs, _ = reach_pairs(O, sub, b)
    hops = {(c, h) for (c, v, h), r in pairs}
    rem = set(O.logs[b - 1]["rem"])
    if rem:
        for c, v in sub.items():
            for h, hop in enumerate(O.trace(c, v)):
                if any(x in rem for x in hop["rows"]):
                    hops.add((c, h))
    if not hops:
        return held
    hops = sorted(hops)
    Q = torch.stack([O.qemb[O.trace(c, held[c])[h]["subq"]] for c, h in hops])
    s = Q @ O.E.T
    s[:, ~O.alive[b]] = -1e4
    sc, top = torch.topk(s, O.k, dim=1)
    by = collections.defaultdict(list)
    for (c, h), rr, ss in zip(hops, top.tolist(), sc.tolist()):
        by[c].append((h, tuple(rr), tuple(float(x) for x in ss)))
    for c, ov in by.items():
        v = held[c]
        prev = dict((h, (r, s_)) for h, r, s_ in v[2]) if isinstance(v, tuple) else {}
        for h, r, s_ in ov:
            prev[h] = (r, s_)
        held[c] = ("R", O.vi(v), tuple(sorted((h, r, s_) for h, (r, s_) in prev.items())))
    return held


def replay(O, policy, stream, scorer=None, lam=None, use_reach=True, qsim=None, ttl=None, verify=None, certify=False,
           refresh=True, fc=None):
    held = {cid: 0 for cid in O.cids}
    suspect = {}
    age = {cid: 0 for cid in O.cids}
    m = collections.Counter(); wtime = 0.0; per_kind = collections.Counter()
    for b in range(1, O.nB + 1):
        cur = {cid: O.cur_vidx(cid, b) for cid in O.cids}
        t = time.time()
        inval = set()
        if policy == "eager_all":
            inval = {c for c, v in held.items() if v is not None}
        elif policy == "ttl":
            inval = {c for c, v in held.items() if v is not None and b - age[c] >= ttl}
        elif policy in ("provenance", "trace_exact", "learned", "qsim", "entity"):
            if policy == "provenance":
                _, prov = reach_pairs(O, {c: v for c, v in held.items() if v is not None}, b); inval = prov
            else:
                sc = entry_scores(O, {c: v for c, v in held.items() if v is not None}, b,
                                  None if policy == "trace_exact" else scorer, use_reach,
                                  qsim if policy == "qsim" else None, policy == "entity")
                thr = 0.0 if policy == "entity" else (lam if lam is not None else -np.inf)
                inval = {c for c, s in sc.items() if s >= thr and s > -np.inf}
        wtime += time.time() - t
        # invalidation quality w.r.t. truth after this batch
        for c, v in held.items():
            if v is None:
                continue
            fresh_before = equiv(O.ans(c, v), O.ans(c, O.cur_vidx(c, b - 1)))
            stale_after = not equiv(O.ans(c, v), O.ans(c, cur[c]))
            if c in inval:
                m["inval"] += 1
                m["inval_tp" if stale_after else "inval_fp"] += 1
            elif stale_after and fresh_before:
                m["inval_fn"] += 1
        for c in inval:
            if certify and held[c] is not None:
                suspect[c] = held[c]
            held[c] = None
        if refresh and policy == "learned":
            held = refresh_states(O, held, b, [c for c, v in held.items() if v is not None])
        if policy == "freshcache":                       # age-gated reuse (FreshCache-style): P(stale | age) <= budget
            for c, v in held.items():
                if v is not None and fc(b - age[c], len(O.trace(c, v))) > lam:
                    held[c] = None
        # queries after batch b
        vt = 0.0
        for c in stream[b - 1]:
            m["q"] += 1
            if held[c] is None and c in suspect:
                cost, ok = certify_cost(O, c, O.vi(suspect.pop(c)), cur[c])   # re-read every hop changed since the last trusted computation
                held[c] = cur[c]; age[c] = b
                m["llm"] += cost; m["cert"] += 1
                if ok:
                    m["hit"] += 1; m["cert_ok"] += 1
                else:
                    m["miss"] += 1
                continue
            if held[c] is None:
                held[c] = cur[c]; age[c] = b
                m["miss"] += 1; m["llm"] += O.calls(c, cur[c])
                continue
            if policy == "judge_on_hit":                  # GroundedCache with a reader-judge: re-retrieve and re-read every hop
                m["llm"] += len(O.trace(c, held[c])); m["verify"] += 1
                if not equiv(O.ans(c, held[c]), O.ans(c, cur[c])):
                    held[c] = cur[c]; age[c] = b; m["miss"] += 1; m["llm"] += O.calls(c, cur[c]); continue
            if policy == "verify_on_hit":
                ok, dt = verify(O, c, held[c], b); vt += dt; m["verify"] += 1
                if not ok:
                    held[c] = cur[c]; age[c] = b
                    m["miss"] += 1; m["llm"] += O.calls(c, cur[c]); m["verify_reject"] += 1
                    continue
            m["hit"] += 1
            if not equiv(O.ans(c, held[c]), O.ans(c, cur[c])):
                m["stale"] += 1
                per_kind["stale"] += 1
        m["verify_time"] += vt
    n_upd = sum(len(lg["events"]) for lg in O.logs)
    res = {"queries": m["q"], "hit_rate": m["hit"] / m["q"], "SRR": m["stale"] / m["q"], "FH": m["stale"] / max(1, m["hit"]),
           "recomputes": m["miss"], "llm_calls": m["llm"], "llm_per_update": m["llm"] / n_upd,
           "inval": m["inval"], "inval_P": m["inval_tp"] / max(1, m["inval"]),
           "inval_R": m["inval_tp"] / max(1, m["inval_tp"] + m["inval_fn"]),
           "write_ms_per_batch": 1000 * wtime / O.nB, "cert": m["cert"], "cert_ok": m["cert_ok"], "cert_per_q": m["cert"] / m["q"], "verify": m["verify"], "verify_ms": 1000 * m["verify_time"] / max(1, m["verify"]),
           "n_updates": n_upd}
    return res


class Verifier:
    """GroundedCache-style read-path validation, made trace-aware (re-retrieves every hop of the cached trace):
    G2 evidence Jaccard >= tau_e, G3 version match on shared docs, G4 lexical support of each hop answer >= tau_s."""

    def __init__(self, O, tau_e=0.5, tau_s=0.6):
        self.O, self.tau_e, self.tau_s = O, tau_e, tau_s
        self.memo = {}

    def __call__(self, O, cid, vidx, b):
        key = (cid, vidx, b)
        t = time.time()
        if key not in self.memo:
            ok = True
            for hop in O.trace(cid, vidx):
                q = O.qemb[hop["subq"]]
                s = q @ O.E.T
                s[~O.alive[b]] = -1e4
                rows = torch.topk(s, O.k).indices.tolist()
                docs_new = {O.doc_of_row[r] for r in rows}; docs_old = set(hop["docs"])
                jac = len(docs_new & docs_old) / max(1, len(docs_new | docs_old))
                same_ver = all(r in hop["rows"] for r in rows if O.doc_of_row[r] in docs_old)
                at = content_tokens(hop["ans"])
                ev = set().union(*[content_tokens(O.texts[r]) for r in rows])
                sup = len(at & ev) / max(1, len(at)) if at else 1.0
                if jac < self.tau_e or not same_ver or sup < self.tau_s:
                    ok = False; break
            self.memo[key] = ok
        return self.memo[key], time.time() - t
