"""Write-path scalability of the Delta-Reader (vectorised), next to the v2 numbers.

For a batch of 20 insertions against H cached hops: (1) encode the inserted documents' claim relations,
(2) reverse reachability GEMM (Thm. 1), (3) exact post-edit evidence sets K' = top-k(K + reaching docs) from the
stored hop scores -- no corpus search, (4) Delta-Reader defeat logits over cached claim relations of K and the
new documents.  Pairwise NLI / set-level cross-encoder costs are extrapolated from measured per-pair time.
"""
import argparse, time
import numpy as np
import torch.nn.functional as F
from common import *
from simulate import Oracle, NLI
from deltar import load_dr

ap = argparse.ArgumentParser()
ap.add_argument("--wl", default="wiki_cf")
args = ap.parse_args()
O = Oracle(args.wl, "test")
OUT = f"{RUNS}/bench"; os.makedirs(OUT, exist_ok=True)
res = {"scal": []}
torch.manual_seed(0)
hops = [(c, 0, h) for c in O.cids for h in range(len(O.trace(c, 0)))]
tr = [O.trace(c, v)[h] for c, v, h in hops]
base_q = torch.stack([O.qemb[t["subq"]] for t in tr])
base_tau = torch.tensor([t["tau"] for t in tr], device="cuda")
base_sc = torch.tensor([t["scores"] for t in tr], device="cuda")
rows = O.ins_rows[0][:20] or list(range(20))
docs = [O.texts[r] for r in rows]
dr = load_dr("full", "_rr")
sx = load_dr("setcross") if os.path.exists(f"{RUNS}/models/dr_setcross.pt") else None
import deltar
sx2 = load_dr("setcross1", "_m3") if os.path.exists(f"{RUNS}/models/dr_setcross1_m3.pt") else None
deltar.DOC_LEN = 160
nli = NLI()

# claim relations of every document cited by a cached hop: computed at index time (not on the write path)
cited = sorted({x for t in tr for x in t["rows"]})
pos = {x: i for i, x in enumerate(cited)}
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    S_l, V_l, M_l, P_l = [], [], [], []
    for i in range(0, len(cited), 128):
        S, V, M, P = dr.doc_states([O.texts[x] for x in cited[i:i + 128]])
        pad = 160 - S.shape[1]
        S_l.append(F.pad(S, (0, 0, 0, pad))); V_l.append(F.pad(V, (0, 0, 0, pad))); M_l.append(F.pad(M.to(torch.uint8), (0, pad)).bool()); P_l.append(P)
    Sb, Vb, Mb, Pb = torch.cat(S_l), torch.cat(V_l), torch.cat(M_l), torch.cat(P_l)
    sig, nu = [], []
    for i in range(0, len(tr), 256):
        a, b = dr.hop_states([t["subq"] for t in tr[i:i + 256]], [t["ans"] for t in tr[i:i + 256]]); sig.append(a); nu.append(b)
    sig, nu = torch.cat(sig), torch.cat(nu)
base_K = torch.tensor([[pos[x] for x in t["rows"]] for t in tr], device="cuda")
Ecorp = O.E[O.alive[0]].half()
res["claim_bytes_per_doc_fp16"] = float((Sb.numel() + Vb.numel()) * 2 / len(cited))
res["claim_bytes_per_doc_int8"] = float((Sb.numel() + Vb.numel()) / len(cited))
res["cited_docs"] = len(cited); res["cached_hops"] = len(hops)
print("claims", res["claim_bytes_per_doc_fp16"], len(cited), len(hops), flush=True)


def timeit(f, rep=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep):
        f()
    torch.cuda.synchronize(); return (time.time() - t) / rep * 1000


# DEFT-X (reads q, a, K') timed once per hop, then released to keep GPU memory for the large-H loop
X_PER_HOP = float("nan")
if sx2 is not None:
    _q = [t["subq"] for t in tr][:64]; _a = [t["ans"] for t in tr][:64]; _k = [[O.texts[x] for x in t["rows"]] for t in tr[:64]]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        X_PER_HOP = timeit(lambda: sx2.setcross_logits(_q, _a, _k), 2) / 64
    del sx2; torch.cuda.empty_cache()
for H in [1000, 10000, 100000, 1000000]:
    rep = (H + len(hops) - 1) // len(hops)
    Q = (base_q.repeat(rep, 1)[:H] + 0.01 * torch.randn(H, base_q.shape[1], device="cuda")).half()
    T = base_tau.repeat(rep)[:H]; SC = base_sc.repeat(rep, 1)[:H]
    idx_hop = torch.arange(H, device="cuda") % len(hops)
    D = O.E[rows].half()
    r = {"H": H}

    @torch.no_grad()
    def write_path():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            Sn, Vn, Mn, Pn = dr.doc_states(docs)                           # (20,T,dim) one encoder pass
            pad = 160 - Sn.shape[1]
            Sn, Vn, Mn = F.pad(Sn, (0, 0, 0, pad)), F.pad(Vn, (0, 0, 0, pad)), F.pad(Mn.to(torch.uint8), (0, pad)).bool()
            s_new = (Q @ D.T).float()                                      # (H,20)
            hit = (s_new >= T[:, None]).any(1).nonzero().squeeze(1)        # reached hops (Thm. 1)
            out = []
            for i in range(0, len(hit), 2048):
                h = hit[i:i + 2048]; hb = idx_hop[h]
                sn = s_new[h]
                top_new = torch.topk(sn, 5, dim=1)                         # at most k new docs can enter
                cand = torch.cat([SC[h], top_new.values], 1)               # (n,10)
                keep = torch.zeros_like(cand, dtype=torch.bool).scatter_(1, torch.topk(cand, 5, dim=1).indices, True)
                Kd = base_K[hb]                                            # (n,5) indices into the claim bank
                S = torch.cat([Sb[Kd], Sn[top_new.indices]], 1); V = torch.cat([Vb[Kd], Vn[top_new.indices]], 1)
                M = torch.cat([Mb[Kd], Mn[top_new.indices]], 1); P = torch.cat([Pb[Kd], Pn[top_new.indices]], 1)
                inK = torch.zeros_like(keep); inK[:, :5] = True
                z, _ = dr.logit(sig[hb], nu[hb], S, V, M & torch.ones_like(M), P, inK, keep)
                out.append(z)
            return out
    r["dr_ms"] = timeit(write_path)
    # latency distribution over repeated batches
    lat = []
    for _ in range(20):
        torch.cuda.synchronize(); t0 = time.time(); write_path(); torch.cuda.synchronize(); lat.append(1000 * (time.time() - t0))
    r["dr_p50_ms"], r["dr_p95_ms"] = float(np.percentile(lat, 50)), float(np.percentile(lat, 95))
    # mixed batch: 4 of the 20 events are deletions of cited documents -> touched hops need one local top-k search
    del_rows = cited[:4]
    hit_del = torch.isin(base_K[idx_hop], torch.tensor([pos[x] for x in del_rows], device="cuda")).any(1).nonzero().squeeze(1)
    @torch.no_grad()
    def deletion_part():
        for i in range(0, len(hit_del), 8192):
            h = hit_del[i:i + 8192]
            torch.topk(Q[h] @ Ecorp.T, 5, dim=1)
    r["del_touched_hops"] = int(len(hit_del))
    r["dr_mixed_ms"] = r["dr_ms"] + (timeit(deletion_part) if len(hit_del) else 0.0)
    r["reach_hops"] = int(((Q @ D.T) >= T[:, None]).any(1).sum())
    r["reach_pairs"] = int(((Q @ D.T) >= T[:, None]).sum())
    subq = [t["subq"] for t in tr][:64]; ans = [t["ans"] for t in tr][:64]
    t = timeit(lambda: nli((docs * 64)[:64], [f"The answer to the question '{q}' is {a}." for q, a in zip(subq, ans)]), 2)
    r["nli_ms"] = t / 64 * r["reach_pairs"]
    if sx is not None:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            t = timeit(lambda: sx.setcross_logits(subq, ans, [[O.texts[x] for x in tt["rows"]] for tt in tr[:64]]), 2)
        r["setcross_ms"] = t / 64 * r["reach_hops"]
    r["xreader_ms"] = X_PER_HOP * r["reach_hops"]
    print(r, flush=True)
    res["scal"].append(r)
    jdump(res, f"{OUT}/bench3_{args.wl}.json")
