"""Write path at scale WITHOUT key replication: H independent cached hops, each citing k distinct documents from a
bank of U unique documents (sharing ratio U / (k H)).  Claim relations live in pinned HOST memory (int8) and only the
documents of touched hops are gathered to the GPU, so the bank need not be GPU resident.

The touched rate per hop and per batch is taken from the real workload (reach on the test stream), not from copies.
Reports: GPU key memory, host claim memory, reach GEMM, host->GPU gather, Delta-Reader scoring, end-to-end p50/p95.
"""
import argparse, time
import numpy as np
import torch.nn.functional as F
from common import *
from simulate import Oracle, reach_pairs
from deltar import load_dr

ap = argparse.ArgumentParser()
ap.add_argument("--wl", default="wiki_cf")
ap.add_argument("--max_host_gb", type=float, default=24)
args = ap.parse_args()
O = Oracle(args.wl, "test")
dr = load_dr("full", "_rr")
torch.manual_seed(0); np.random.seed(0)
K = O.k
# real touched rate: touched hops per cached hop per update batch
nh = tot = 0
for b in range(1, O.nB + 1):
    held = {c: O.cur_vidx(c, b - 1) for c in O.cids}
    pairs, _ = reach_pairs(O, held, b)
    tot += len({(c, h) for (c, v, h), r in pairs}); nh += sum(len(O.trace(c, v)) for c, v in held.items() if v is not None)
rate = tot / nh
docs = [O.texts[r] for r in (O.ins_rows[0][:20] or list(range(20)))]
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    S0, V0, M0, P0 = dr.doc_states(docs[:4])
    t0 = O.trace(O.cids[0], 0)[0]
    sg0, nu0 = dr.hop_states([t0["subq"]], [t0["ans"]])
T_, ds, dv = 160, S0.shape[-1], V0.shape[-1]
P_shape = P0.shape[1:]
bytes_doc = T_ * (ds + dv)                                  # int8 claim relations per document
res = {"wl": args.wl, "touch_rate_per_hop_batch": rate, "claim_bytes_per_doc_int8": bytes_doc, "rows": []}
print(res, flush=True)


def timeit(f, rep=10):
    f(); torch.cuda.synchronize(); lat = []
    for _ in range(rep):
        torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize(); lat.append(1000 * (time.time() - t))
    return float(np.percentile(lat, 50)), float(np.percentile(lat, 95))


for H in [100000, 1000000]:
    for share in [1.0, 0.2]:
        U = int(K * H * share)
        host_gb = U * bytes_doc / 2 ** 30
        U_alloc = U if host_gb <= args.max_host_gb else int(args.max_host_gb * 2 ** 30 / bytes_doc)
        bankS = torch.randint(-127, 127, (U_alloc, T_, ds), dtype=torch.int8).pin_memory()
        bankV = torch.randint(-127, 127, (U_alloc, T_, dv), dtype=torch.int8).pin_memory()
        Q = F.normalize(torch.randn(H, O.E.shape[1], device="cuda"), dim=1).half()
        n_touch = max(1, int(round(rate * H)))
        tau = torch.full((H,), 10.0, device="cuda"); tau[torch.randperm(H, device="cuda")[:n_touch]] = -10.0  # real touched count
        Kidx = torch.randint(0, U_alloc, (H, K))
        sig = sg0.expand(H, -1).contiguous(); nu = nu0.expand(H, -1).contiguous()
        D = F.normalize(torch.randn(20, O.E.shape[1], device="cuda"), dim=1).half()
        r = {"H": H, "share": share, "unique_docs": U, "host_claim_GiB": host_gb, "gpu_key_MiB": H * (O.E.shape[1] + 1) * 2 / 2 ** 20,
             "touched_hops": n_touch}

        @torch.no_grad()
        def reach():
            return ((Q @ D.T).float() >= tau[:, None]).any(1).nonzero().squeeze(1)
        hit = reach()
        rows = Kidx[hit.cpu()].reshape(-1)
        uniq, inv = torch.unique(rows, return_inverse=True)
        r["gathered_docs"] = int(len(uniq)); r["gathered_MiB"] = len(uniq) * bytes_doc / 2 ** 20

        bufS = torch.empty((len(uniq), T_, ds), dtype=torch.int8).pin_memory()
        bufV = torch.empty((len(uniq), T_, dv), dtype=torch.int8).pin_memory()

        @torch.no_grad()
        def gather():                                  # gather into a pinned staging buffer, then one async DMA
            torch.index_select(bankS, 0, uniq, out=bufS); torch.index_select(bankV, 0, uniq, out=bufV)
            return bufS.cuda(non_blocking=True), bufV.cuda(non_blocking=True)
        gS, gV = gather()

        @torch.no_grad()
        def score():
            out = []
            with torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(0, len(hit), 2048):
                    h = hit[i:i + 2048]; n = len(h)
                    ix = inv.view(-1, K)[i:i + n].cuda()
                    S = torch.cat([gS[ix], gS[ix[:, :1]]], 1).float() / 127; V = torch.cat([gV[ix], gV[ix[:, :1]]], 1).float() / 127
                    M = torch.ones(n, K + 1, T_, dtype=torch.bool, device="cuda")
                    P = torch.zeros(n, K + 1, device="cuda")
                    inK = torch.zeros(n, K + 1, dtype=torch.bool, device="cuda"); inK[:, :K] = True
                    keep = torch.ones_like(inK); keep[:, 0] = False
                    z, _ = dr.logit(sig[h], nu[h], S, V, M, P, inK, keep)
                    out.append(z)
            return out
        try:
            score()
            r["reach_ms_p50"], _ = timeit(reach)
            r["gather_ms_p50"], r["gather_ms_p95"] = timeit(gather)
            r["score_ms_p50"], r["score_ms_p95"] = timeit(score)
            r["e2e_ms_p50"], r["e2e_ms_p95"] = timeit(lambda: (reach(), gather(), score()))
        except Exception as e:
            r["error"] = str(e)[:200]
        print(r, flush=True); res["rows"].append(r)
        jdump(res, f"{RUNS}/bench/bench_indep_{args.wl}.json")
        del bankS, bankV, bufS, bufV, Q, gS, gV; torch.cuda.empty_cache()
print("done", flush=True)
