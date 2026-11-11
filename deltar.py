"""Delta-Reader: a differentiable surrogate of the RAG reader that predicts whether an *evidence-set edit*
changes a hop answer.

By Trace Invariance (Thm. 1) a hop answer depends on the corpus only through its top-k evidence set K.  Every
corpus update (insert / delete / modify) therefore acts on a cached hop as an exact, geometrically computable
edit K -> K'.  The Delta-Reader sees exactly this sufficient statistic -- (sub-question q, cached hop answer a,
K, K') -- and nothing else.

Documents are treated as small *relations* of (slot, value) claims rather than single tuples: every token of a
document carries a slot key s_t and a value vector v_t.  For a hop with slot query sigma(q) and cached value
nu(a), a document d is read *conditionally on the slot*:
    r(d)  = (1/beta) logsumexp_t beta <sigma, s_t>                      (does d speak about this slot?)
    v(d)  = sum_t softmax_t(beta <sigma, s_t>) v_t                      (which value does d assert for it?)
    g(d)  = cos(nu, v(d))                                               (agreement with the cached answer)
An evidence set is aggregated with reader-like attention  w_d = softmax_d(gamma r(d) + p(d))  (p = learned
document prior, e.g. recency), giving the support of the cached answer  A(K) = sum_d w_d g(d).
Defeat logit of the edit:
    z = k1 (A(K) - A(K'))  +  k2 (1 - A(K'))  +  c
(belief shift + residual support).  Rebuttal (a competing claim enters), undercutting (the supporting document is
evicted or deleted) and fragility (A(K) already low) are all special cases of one quantity.

Training = counterfactual distillation of the real reader: labels are the reader's own answer under K' (from the
oracle replay of the training streams, and from single-document counterfactual reads).  Auxiliary grounding:
the slot-conditioned value v(d) is aligned with the reader's answer when reading d alone (InfoNCE) and r(d) with
whether d answers the slot at all (BCE).

Variants: full | noset (sees only the documents that enter K) | doclevel (one claim per document, i.e. a
slot-independent value) | noaux | setcross (cross-encoder over (q, a, K'), expensive reference).
"""
import argparse, random, math, collections
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from common import *

DOC_LEN = int(os.environ.get("DR_DOCLEN", 160))
SET_LEN = int(os.environ.get("DR_SETLEN", 1024))


class DeltaReader(nn.Module):
    def __init__(self, path, dim=128, variant="full"):
        super().__init__()
        self.variant = variant
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path)
        H = self.enc.config.hidden_size
        if variant.startswith("setcross"):
            self.head = nn.Linear(H, 1)
        elif variant.startswith("mtr"):                # margin-transport reader: shift and scale of the post-edit margin
            self.head = nn.Linear(H, 2)
            self.log_kappa = nn.Parameter(torch.tensor(0.0))
        else:
            self.Ps = nn.Linear(H, dim)
            self.Pv = nn.Linear(H, dim)
            self.Pp = nn.Linear(H, 1)
        self.log_beta = nn.Parameter(torch.tensor(math.log(10.0)))
        self.log_gamma = nn.Parameter(torch.tensor(math.log(10.0)))
        self.k1 = nn.Parameter(torch.tensor(4.0)); self.k2 = nn.Parameter(torch.tensor(4.0)); self.c = nn.Parameter(torch.tensor(0.0))
        self.a_r = nn.Parameter(torch.tensor(8.0)); self.c_r = nn.Parameter(torch.tensor(-4.0))
        self.a_g = nn.Parameter(torch.tensor(4.0)); self.c_g = nn.Parameter(torch.tensor(0.0))
        self.t_v = nn.Parameter(torch.tensor(math.log(20.0)))

    @property
    def dev(self):
        return next(self.parameters()).device

    # ------------------------------------------------------------ representations
    def hop_states(self, subqs, answers):
        b = self.tok([QPREFIX + q for q in subqs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.dev)
        hs = self.enc(**b).last_hidden_state[:, 0]
        b = self.tok(list(answers), padding=True, truncation=True, max_length=24, return_tensors="pt").to(self.dev)
        ha = self.enc(**b).last_hidden_state[:, 0]
        return F.normalize(self.Ps(hs), dim=-1), F.normalize(self.Pv(ha), dim=-1)

    def doc_states(self, docs):
        """per-document claim relation: slot keys S (n,T,dim), values V (n,T,dim), token mask (n,T), prior p (n,)"""
        b = self.tok(list(docs), padding=True, truncation=True, max_length=DOC_LEN, return_tensors="pt").to(self.dev)
        h = self.enc(**b).last_hidden_state
        p = self.Pp(h[:, 0]).squeeze(-1)
        if self.variant == "doclevel":
            h = h[:, :1]; m = torch.ones(h.shape[0], 1, dtype=torch.bool, device=self.dev)
        else:
            m = b["attention_mask"].bool().clone(); m[:, 0] = False          # tokens are the claims
        return F.normalize(self.Ps(h), dim=-1), F.normalize(self.Pv(h), dim=-1), m, p

    def read(self, sigma, nu, S, V, M):
        """slot-conditioned reading of documents.  sigma, nu: (B,dim); S,V: (B,D,T,dim); M: (B,D,T) -> r, g, v (B,D[,dim])"""
        beta = self.log_beta.exp()
        att = beta * torch.einsum("bd,bmtd->bmt", sigma, S)
        att = att.masked_fill(~M, -1e4)
        r = torch.logsumexp(att, -1) / beta
        w = torch.softmax(att, -1)
        v = F.normalize(torch.einsum("bmt,bmtd->bmd", w, V), dim=-1)
        g = torch.einsum("bd,bmd->bm", nu, v)
        return r, g, v

    def support(self, r, g, p, mask):
        """reader-like aggregation of an evidence set: A = sum_d softmax(gamma r + p) g"""
        lg = (self.log_gamma.exp() * r + p).masked_fill(~mask, -1e4)
        w = torch.softmax(lg, -1) * mask.any(-1, keepdim=True)
        return (w * g).sum(-1)

    def logit(self, sigma, nu, S, V, M, p, in_K, in_K2):
        r, g, v = self.read(sigma, nu, S, V, M)
        A2 = self.support(r, g, p, in_K2)
        if self.variant == "noset":
            A1 = torch.ones_like(A2)
        else:
            A1 = self.support(r, g, p, in_K)
        return self.k1 * (A1 - A2) + self.k2 * (1 - A2) + self.c, (r, g, v)

    def setcross_logits(self, subqs, answers, after_docs, before_docs=None):
        left = [f"Question: {q} Current answer: {a}" for q, a in zip(subqs, answers)]
        if before_docs is None:
            right = [" ".join(ds) for ds in after_docs]
        else:                                          # reads the old and the new evidence set jointly
            right = ["Before: " + " | ".join(b_) + " After: " + " | ".join(a_) for b_, a_ in zip(before_docs, after_docs)]
        b = self.tok(left, right, padding=True, truncation="only_second", max_length=512 if self.variant == "setcross" else SET_LEN,
                     return_tensors="pt").to(self.dev)
        return self.head(self.enc(**b).last_hidden_state[:, 0]).squeeze(-1)


# ------------------------------------------------------------------ batching helpers
def gather_docs(model, examples, cache=None, texts_of=None):
    """encode the union of documents used by a list of examples; returns per-example tensors (B,D,...) and masks."""
    keys = []
    for ex in examples:
        for d in ex["Kset"] + ex["K2only"]:
            keys.append(d)
    uniq = list(dict.fromkeys(keys))
    if cache is not None:
        need = [d for d in uniq if d not in cache]
        for i in range(0, len(need), 96):
            ch = need[i:i + 96]
            S, V, M, p = model.doc_states([texts_of(d) for d in ch])
            if os.environ.get("DR_INT8"):              # storage study: claim vectors quantised to int8 (symmetric, per vector)
                q8 = lambda X: (lambda m_: torch.round(X.float() / m_ * 127) * m_ / 127)(X.float().abs().amax(-1, keepdim=True).clamp_min(1e-8))
                S, V = q8(S), q8(V)
            for j, d in enumerate(ch):
                cache[d] = (S[j], V[j], M[j], p[j])
        S = torch.nn.utils.rnn.pad_sequence([cache[d][0] for d in uniq], batch_first=True)
        V = torch.nn.utils.rnn.pad_sequence([cache[d][1] for d in uniq], batch_first=True)
        M = torch.nn.utils.rnn.pad_sequence([cache[d][2].to(torch.uint8) for d in uniq], batch_first=True).bool()
        P = torch.stack([cache[d][3] for d in uniq])
    else:
        S, V, M, P = model.doc_states([texts_of(d) for d in uniq])
    pos = {d: i for i, d in enumerate(uniq)}
    Dmax = max(len(ex["Kset"]) + len(ex["K2only"]) for ex in examples)
    B = len(examples)
    idx = torch.zeros(B, Dmax, dtype=torch.long, device=S.device)
    inK = torch.zeros(B, Dmax, dtype=torch.bool, device=S.device)
    inK2 = torch.zeros(B, Dmax, dtype=torch.bool, device=S.device)
    valid = torch.zeros(B, Dmax, dtype=torch.bool, device=S.device)
    for i, ex in enumerate(examples):
        docs = ex["Kset"] + ex["K2only"]
        for m, d in enumerate(docs):
            idx[i, m] = pos[d]; valid[i, m] = True
            inK[i, m] = m < len(ex["Kset"])
            inK2[i, m] = d in ex["K2set"]
    return S[idx], V[idx], M[idx] & valid.unsqueeze(-1), P[idx], inK, inK2, idx, pos


def prep(ex, variant):
    """normalise an example into K list, entering docs, and the K' membership set"""
    K, K2 = list(ex["K"]), list(ex["K2"])
    Kset = list(dict.fromkeys(K))
    K2only = [d for d in dict.fromkeys(K2) if d not in set(Kset)]
    K2set = set(K2)
    if variant == "noset":
        ent = K2only if K2only else list(dict.fromkeys(K2))
        Kset, K2only, K2set = [], ent, set(ent)
    return {**ex, "Kset": Kset, "K2only": K2only, "K2set": K2set}


def score_examples(model, examples, cache, texts_of, bs=64):
    """defeat logits for a list of examples (no grad); doc states are cached across calls"""
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(0, len(examples), bs):
            B = [prep(e, model.variant) for e in examples[i:i + bs]]
            if model.variant.startswith("setcross"):
                out.append(cross_logits(model, B, texts_of).float())
                continue
            if model.variant.startswith("mtr"):
                m_hat, sc = mtr_forward(model, B, texts_of)
                out.append((-m_hat / sc).float())
                continue
            sig, nu = model.hop_states([e["subq"] for e in B], [e["ans"] for e in B])
            S, V, M, P, inK, inK2, _, _ = gather_docs(model, B, cache, texts_of)
            z, _ = model.logit(sig, nu, S, V, M, P, inK, inK2)
            out.append(z.float())
    return torch.cat(out).cpu().numpy() if out else np.zeros(0)


# ------------------------------------------------------------------ training data
def version_before(R, cid, b):
    v = None
    for vb, r in R["versions"][cid]:
        if vb < b:
            v = r
    return v


def edit_examples(wl, split="train"):
    """hop-level evidence edits observed in an oracle replay; label = the reader's hop answer under K' changed."""
    R = pload(f"{RUNS}/{wl}/{split}/oracle.pkl")
    exs = []
    for lg in R["logs"]:
        b = lg["b"]
        for cid in lg["reach"]:
            new = [r for vb, r in R["versions"][cid] if vb == b]
            old = version_before(R, cid, b)
            if not new or old is None:
                continue
            for ho, hn in zip(old["trace"], new[0]["trace"]):
                if ho["subq"] != hn["subq"]:
                    break
                if ho["rows"] == hn["rows"]:
                    continue
                y = int(not equiv(ho["ans"], hn["ans"]))
                exs.append({"wl": wl, "src": "edit", "subq": ho["subq"], "ans": ho["ans"],
                            "K": [(wl, x) for x in ho["rows"]], "K2": [(wl, x) for x in hn["rows"]], "y": y})
                if ho["ans"] != hn["ans"]:
                    break
    return exs, R["texts"]


def pair_examples(wl, R_texts):
    """single-document counterfactual reads (label.py): K' = top-k(K + {d}); also the alone-reading of d."""
    p = f"{RUNS}/{wl}/train/pairs.json"
    if not os.path.exists(p):
        return []
    R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
    k = R["k"]
    E = torch.tensor(R["E"]).float()
    exs = []
    for r in jload(p):
        if not r["reach"]:
            continue
        hop = version_before(R, r["cid"], r["b"])["trace"][r["h"]]
        s_new = float(torch.tensor(R["qemb"][hop["subq"]]).float() @ E[r["row"]])
        cand = sorted(list(zip(hop["scores"], hop["rows"])) + [(s_new, r["row"])], key=lambda x: -x[0])[:k]
        exs.append({"wl": wl, "src": "pair", "subq": hop["subq"], "ans": hop["ans"],
                    "K": [(wl, x) for x in hop["rows"]], "K2": [(wl, x) for _, x in cand],
                    "y": int(not equiv(r["cf_ans"], hop["ans"])), "doc": (wl, r["row"]), "r": r["r"], "read": r["read"]})
    return exs


def cross_logits(model, B, texts_of):
    """cross-attention readers: setcross (K', 512 tok.), setcross1 (K'), setcrossE (only documents entering K'),
    setcross2 (K and K' jointly: the differential cross-reader)"""
    if model.variant in ("setcrossE", "setcrossD", "mtrE"):
        after = [[texts_of(d) for d in (e["K2"] if not [x for x in e["K2"] if x not in set(e["K"])] else [x for x in e["K2"] if x not in set(e["K"])])] for e in B]
    else:
        after = [[texts_of(d) for d in e["K2"]] for e in B]
    if model.variant == "setcross2":
        before = [[texts_of(d) for d in e["K"]] for e in B]
    elif model.variant in ("setcrossD", "mtrE"):  # the delta only: documents that left K (old versions) next to those that entered
        before = [[texts_of(d) for d in e["K"] if d not in set(e["K2"])] or ["(none)"] for e in B]
    else:
        before = None
    return model.setcross_logits([e["subq"] for e in B], [e["ans"] for e in B], after, before)


def mtr_forward(model, B, texts_of):
    """margin-transport reader: the post-edit margin of the cached answer is the trusted margin, transported by a
    shift that the cross-reader reads off the edited evidence,  m_hat = kappa * log(1 + mu*) + Delta(q, a, K'),
    with an input-dependent scale s.  P(defeat) = P(m < 0) = sigmoid(-m_hat / s) under a logistic margin law."""
    out = cross_logits(model, B, texts_of).float()
    delta, sc = out[:, 0], F.softplus(out[:, 1]) + 0.05
    if model.variant == "mtr0":                     # ablation: no trusted margin (the shift must carry everything)
        return delta, sc
    mu = torch.tensor([e.get("mu", 0.0) for e in B], dtype=torch.float, device=delta.device)
    return model.log_kappa.exp() * torch.log1p(mu) + delta, sc


def margin_nll(m_hat, sc, B):
    """censored logistic likelihood of the reader's signed post-edit margin (exact, >= value, or <= value)"""
    m2 = torch.tensor([e["m2"] for e in B], dtype=torch.float, device=m_hat.device)
    cen = torch.tensor([e["cen"] for e in B], dtype=torch.float, device=m_hat.device)
    u = (m2 - m_hat) / sc
    exact = -F.logsigmoid(u) - F.logsigmoid(-u) + torch.log(sc)
    right = -F.logsigmoid(-u)                        # m >= m2
    left = -F.logsigmoid(u)                          # m <= m2
    return torch.where(cen == 0, exact, torch.where(cen > 0, right, left)).mean()


def margin_examples(wl, split="train", synth=True):
    """reader-labelled evidence edits with their margins: replay edits, single-document counterfactuals and synthetic
    edits around the training hop states.  Each example carries the trusted margin mu* of the cached answer on K and
    the signed margin m2 of the cached tokens on K' (with its censoring), aligned with answer equivalence: a token
    change to an equivalent answer is not a defeat (m >= 0), a non-equivalent change is (m <= 0)."""
    from mcert import MarginBook, MCAP
    MB = MarginBook(wl, split)
    e1, T = edit_examples(wl, split)
    e2 = pair_examples(wl, T) if split == "train" else []
    e3 = []
    if synth:
        for s in MB.synth:
            a, b = MB.S.get((s["subq"], tuple(s["K"]))), MB.S.get((s["subq"], tuple(s["K2"])))
            if a is None or b is None:
                continue
            e3.append({"wl": wl, "src": "synth", "subq": s["subq"], "ans": a["ans"], "K": [(wl, x) for x in s["K"]],
                       "K2": [(wl, x) for x in s["K2"]], "y": int(not equiv(a["ans"], b["ans"]))})
    out = []
    for e in e1 + e2 + e3:
        q, K, K2 = e["subq"], [x for _, x in e["K"]], [x for _, x in e["K2"]]
        if not MB.has(q, K) or not MB.has(q, K2):
            continue
        m2, c = MB.post(q, K, K2)
        cen = -1 if c else 0
        if m2 >= MCAP - 1e-3:
            m2, cen = MCAP, 1
        if e["y"] == 0 and m2 < 0:
            m2, cen = 0.0, 1
        elif e["y"] == 1 and m2 >= 0:
            m2, cen = 0.0, -1
        out.append({**e, "mu": MB.mu(q, K), "m2": float(m2), "cen": cen})
    print(wl, split, "margin examples: edit", len(e1), "pair", len(e2), "synth", len(e3), "kept", len(out),
          "defeating", sum(e["y"] for e in out), flush=True)
    return out, T


def sup_loss(z, y, pw, B):
    """reader labels where available (y in {0,1}); teacher probabilities (soft labels) where a teacher was used"""
    z = z.float(); lab = y >= 0
    loss = F.binary_cross_entropy_with_logits(z[lab], y[lab], pos_weight=pw) if lab.any() else z.sum() * 0
    if "pt" in B[0]:
        pt = torch.tensor([e["pt"] for e in B], dtype=torch.float, device=z.device)
        loss = loss + F.binary_cross_entropy_with_logits(z, pt)
    return loss


def synth_edits(wl, n_per, rng):
    """unlabelled evidence edits around the training hop states: one eviction (a document leaves K, the next-ranked
    corpus document enters) and one insertion (a near-miss corpus document enters at its rank) per state"""
    R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
    E = torch.tensor(R["E"]).float().cuda(); k = R["k"]
    states = {}
    for cid, vs in R["versions"].items():
        for _, res in vs:
            for hop in res["trace"]:
                states[(hop["subq"], hop["ans"], tuple(hop["rows"]))] = hop
    keys = list(states)
    out = []
    for i in range(0, len(keys), 512):
        ch = keys[i:i + 512]
        Q = torch.stack([torch.tensor(R["qemb"][q]).float().cuda() for q, _, _ in ch])
        sc, top = torch.topk(Q @ E.T, k + 20, dim=1)
        for (q, a, rows), ss, tt in zip(ch, sc.tolist(), top.tolist()):
            cand = [(s_, t_) for s_, t_ in zip(ss, tt) if t_ not in rows]
            if not cand:
                continue
            hop = states[(q, a, rows)]
            K = list(rows)
            j = rng.randrange(len(K))                                       # eviction edit
            K2 = [x for x in K if x != K[j]] + [cand[0][1]]
            out.append({"wl": wl, "src": "synth", "subq": q, "ans": a, "K": [(wl, x) for x in K], "K2": [(wl, x) for x in K2], "y": -1})
            if n_per > 1:                                                    # insertion edit
                s_d, d = cand[rng.randrange(min(len(cand), 10))]
                merged = sorted(list(zip(hop["scores"], K)) + [(max(s_d, hop["tau"] + 1e-4), d)], key=lambda x: -x[0])[:k]
                out.append({"wl": wl, "src": "synth", "subq": q, "ans": a, "K": [(wl, x) for x in K], "K2": [(wl, x) for _, x in merged], "y": -1})
    return out


def add_teacher(exs, args, texts_of):
    rng = random.Random(args.seed)
    for wl in args.wls.split(","):
        sy = synth_edits(wl, args.synth, rng)
        print(wl, "synthetic edits", len(sy), flush=True)
        exs = exs + sy
    tv, tt = (args.teacher.split(":") + [""])[:2]
    global DOC_LEN, SET_LEN
    keep = (DOC_LEN, SET_LEN)
    T = load_dr(tv, tt)
    z = score_examples(T, exs, {}, texts_of, bs=32)
    DOC_LEN, SET_LEN = keep
    del T; torch.cuda.empty_cache()
    for e, x in zip(exs, z):
        e["pt"] = float(1 / (1 + np.exp(-x)))
    print("teacher-labelled examples", len(exs), "mean p", float(np.mean([e["pt"] for e in exs])), flush=True)
    return exs


def is_rel(read):
    return bool(read) and canon(read) != ("<abstain>",)


def grounding_loss(model, B, r, g, v, reads):
    """claim-level distillation: for every document of an example whose alone-read by the reader is known,
    r(d) must predict whether d answers the sub-question, g(d) whether it asserts the cached answer, and the
    slot-conditioned value v(d) must match the value the reader extracts (in-batch InfoNCE)."""
    pos_r, lab_r, pos_g, lab_g, vt_q, vt_a = [], [], [], [], [], []
    for i, e in enumerate(B):
        for m, d in enumerate(e["Kset"] + e["K2only"]):
            rd = reads.get((d[0], e["subq"], d[1]))
            if rd is None and e.get("doc") == d:
                rd = e.get("read")
            if rd is None:
                continue
            rel = is_rel(rd)
            pos_r.append((i, m)); lab_r.append(float(rel))
            if rel:
                pos_g.append((i, m)); lab_g.append(float(equiv(rd, e["ans"]))); vt_q.append(e["subq"]); vt_a.append(rd)
    loss = 0.0
    if pos_r:
        ii = torch.tensor(pos_r, device=r.device)
        loss = loss + F.binary_cross_entropy_with_logits((model.a_r * r[ii[:, 0], ii[:, 1]] + model.c_r).float(), torch.tensor(lab_r, device=r.device))
    if pos_g:
        ii = torch.tensor(pos_g, device=r.device)
        loss = loss + F.binary_cross_entropy_with_logits((model.a_g * g[ii[:, 0], ii[:, 1]] + model.c_g).float(), torch.tensor(lab_g, device=r.device))
        vals = [normalize(a) for a in vt_a]
        uniq = list(dict.fromkeys(vals))
        if len(uniq) > 1:
            first = {u: vt_a[vals.index(u)] for u in uniq}
            vt = model.hop_states(vt_q[:len(uniq)], [first[u] for u in uniq])[1]       # value embeddings of distinct values
            vd = v[ii[:, 0], ii[:, 1]]
            sim = (vd @ vt.T).float() * model.t_v.exp()
            tgt = torch.tensor([uniq.index(u) for u in vals], device=r.device)
            loss = loss + F.cross_entropy(sim, tgt)
    return loss


def load_reads(wls):
    reads = {}
    for wl in wls:
        p = f"{RUNS}/{wl}/train/reads.json"
        if os.path.exists(p):
            for x in jload(p):
                reads[(wl, x["subq"], x["row"])] = x["read"]
    return reads


def train(args):
    torch.manual_seed(args.seed); random.seed(args.seed)
    exs, texts = [], {}
    for wl in args.wls.split(","):
        if args.model.startswith("mtr") or args.aug:     # reader-labelled edits with margins (same data for every variant)
            e, T = margin_examples(wl, synth=not args.nosynth)
            texts[wl] = T; exs += e
            continue
        e1, T = edit_examples(wl)
        texts[wl] = T
        e2 = pair_examples(wl, T)
        print(wl, "edit", len(e1), "pos", sum(e["y"] for e in e1), "pair", len(e2), "pos", sum(e["y"] for e in e2), flush=True)
        exs += e1 + e2
    texts_of = lambda d: texts[d[0]][d[1]]
    reads = load_reads(args.wls.split(",")) if args.reads else {}
    print("claim-level reads", len(reads), flush=True)
    model = DeltaReader(args.backbone or bge_path(), variant=args.model).cuda()
    if args.ckpt:
        model.enc.gradient_checkpointing_enable()
    if args.init:                                     # re-distillation for a new reader: start from a trained model
        model.load_state_dict(torch.load(f"{RUNS}/models/{args.init}.pt", map_location="cuda"), strict=False)
    if args.lowmem:     # shared GPU: frozen word-embedding matrix (bf16) and 8-bit AdamW states
        for n_, p_ in model.enc.named_parameters():
            if "word_embeddings" in n_:
                p_.requires_grad_(False); p_.data = p_.data.to(torch.bfloat16)
    enc_params = [p for p in model.enc.parameters() if p.requires_grad]
    other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    import bitsandbytes as bnb
    opt = (bnb.optim.AdamW8bit if args.lowmem else torch.optim.AdamW)([{"params": enc_params, "lr": args.lr}, {"params": other, "lr": 1e-3}], weight_decay=0.01)
    if args.teacher:
        exs = add_teacher(exs, args, texts_of)
    pos = sum(e["y"] == 1 for e in exs); neg = sum(e["y"] == 0 for e in exs)
    pw = torch.tensor(neg / max(pos, 1), device="cuda")
    print("train examples", len(exs), "pos", pos, "pos_weight", float(pw), flush=True)
    steps = args.epochs * math.ceil(len(exs) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    model.train()
    for ep in range(args.epochs):
        random.shuffle(exs)
        tot = 0.0; nb = 0
        for i in range(0, len(exs), args.bs):
            BB = [prep(e, args.model) for e in exs[i:i + args.bs]]
            opt.zero_grad(); lsum = 0.0
            for j in range(0, len(BB), args.micro):   # gradient accumulation: same effective batch, bounded memory
                B = BB[j:j + args.micro]
                y = torch.tensor([e["y"] for e in B], dtype=torch.float, device="cuda")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if args.model.startswith("mtr"):
                        m_hat, sc = mtr_forward(model, B, texts_of)
                        z = -m_hat / sc
                        loss = margin_nll(m_hat, sc, B)
                    elif args.model.startswith("setcross"):
                        z = cross_logits(model, B, texts_of)
                        loss = sup_loss(z, y, pw, B)
                    else:
                        sig, nu = model.hop_states([e["subq"] for e in B], [e["ans"] for e in B])
                        S, V, M, P, inK, inK2, idx, posmap = gather_docs(model, B, None, texts_of)
                        z, (r, g, v) = model.logit(sig, nu, S, V, M, P, inK, inK2)
                        loss = sup_loss(z, y, pw, B)
                        if args.model != "noaux":
                            loss = loss + grounding_loss(model, B, r, g, v, reads)
                    if args.rank and (y == 1).sum() > 0 and (y == 0).sum() > 0:
                        # conformal control acts on the ordering of scores only: logistic ranking of defeats above non-defeats
                        zp, zn = z.float()[y == 1], z.float()[y == 0]
                        loss = loss + F.softplus(zn[None, :] - zp[:, None]).mean()
                (loss * len(B) / len(BB)).backward(); lsum += float(loss) * len(B) / len(BB)
            loss = torch.tensor(lsum)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += float(loss); nb += 1
            if nb % 200 == 0:
                print(f"  ep {ep} step {nb} loss {tot / nb:.4f}", flush=True)
        print(f"epoch {ep} loss {tot / max(nb, 1):.4f}", flush=True)
    os.makedirs(f"{RUNS}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{RUNS}/models/dr_{args.model}{args.tag}.pt")
    jdump({"backbone": args.backbone or bge_path(), "doclen": DOC_LEN, "setlen": SET_LEN, "wls": args.wls}, f"{RUNS}/models/dr_{args.model}{args.tag}.json")
    print("saved", flush=True)


def load_dr(variant, tag=""):
    global DOC_LEN, SET_LEN
    meta = f"{RUNS}/models/dr_{variant}{tag}.json"
    cfg = jload(meta) if os.path.exists(meta) else {}
    DOC_LEN = cfg.get("doclen", 320 if "320" in tag else DOC_LEN); SET_LEN = cfg.get("setlen", SET_LEN)
    m = DeltaReader(cfg.get("backbone", bge_path()), variant=variant).cuda().eval()
    m.load_state_dict(torch.load(f"{RUNS}/models/dr_{variant}{tag}.pt", map_location="cuda"), strict=False)
    if os.environ.get("DR_BF16"):                      # shared GPU: bf16 weights (inference runs under bf16 autocast anyway)
        m = m.to(torch.bfloat16)
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="full")
    ap.add_argument("--wls", default="wiki_cf,hotpot,wiki_t")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lowmem", action="store_true", help="frozen word embeddings + 8-bit AdamW")
    ap.add_argument("--micro", type=int, default=16, help="micro-batch for gradient accumulation")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--reads", action="store_true")
    ap.add_argument("--rank", action="store_true")
    ap.add_argument("--backbone", default="")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (large backbones)")
    ap.add_argument("--init", default="")
    ap.add_argument("--teacher", default="", help="variant:tag of a differential cross-reader to distil from")
    ap.add_argument("--synth", type=int, default=2, help="synthetic teacher-labelled edits per training hop state")
    ap.add_argument("--aug", action="store_true", help="train on the reader-labelled edits of margin_examples (synthetic included)")
    ap.add_argument("--nosynth", action="store_true", help="margin examples without the synthetic edits")
    train(ap.parse_args())
