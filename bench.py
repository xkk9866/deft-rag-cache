"""Write-path scalability and read-path latency micro-benchmarks.

(1) per-batch invalidation cost vs cache size H (hop keys): eager re-retrieval, trace-exact reverse GEMM,
    DEFT (probe encoding + fused GEMM), NLI / cross-encoder on reachable pairs.
(2) single-request latency of the RAG pipeline (cache miss), of a cache hit, and of read-path verification.
"""
import argparse, time
import numpy as np
from common import *
from simulate import Oracle, NLI
from deft import load_model

ap = argparse.ArgumentParser()
ap.add_argument("--wl", default="wiki_cf")
ap.add_argument("--latency", action="store_true")
args = ap.parse_args()
O = Oracle(args.wl, "test")
OUT = f"{RUNS}/bench"; os.makedirs(OUT, exist_ok=True)
res = {"scal": [], "lat": {}}
torch.manual_seed(0)
hops = [(c, 0, h) for c in O.cids for h in range(len(O.trace(c, 0)))]
base_q = torch.stack([O.qemb[O.trace(c, v)[h]["subq"]] for c, v, h in hops])
base_tau = torch.tensor([O.trace(c, v)[h]["tau"] for c, v, h in hops], device="cuda")
rows = O.ins_rows[0][:20] or list(range(20))
docs = [O.texts[r] for r in rows]
svde = load_model("svdef"); cross = load_model("cross") if os.path.exists(f"{RUNS}/models/cross.pt") else None
nli = NLI()
Kd = svde.key_vectors([O.trace(c, v)[h]["subq"] for c, v, h in hops], [O.trace(c, v)[h]["ans"] for c, v, h in hops])
Ecorp = O.E[O.alive[0]].half()


def timeit(f, rep=3):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(rep):
        f()
    torch.cuda.synchronize(); return (time.time() - t) / rep * 1000


for H in [1000, 10000, 100000, 1000000]:
    rep = (H + len(hops) - 1) // len(hops)
    Q = (base_q.repeat(rep, 1)[:H] + 0.01 * torch.randn(H, base_q.shape[1], device="cuda")).half()
    T = base_tau.repeat(rep)[:H]
    K = Kd.repeat(rep, 1)[:H].half()
    D = O.E[rows].half()
    r = {"H": H}
    # eager forward re-retrieval of every cached hop (chunked top-k over the corpus)
    def eager():
        for i in range(0, H, 8192):
            torch.topk(Q[i:i + 8192] @ Ecorp.T, O.k, dim=1)
    r["eager_reretrieve_ms"] = timeit(eager, 1) if H <= 100000 else timeit(eager, 1)
    r["trace_exact_ms"] = timeit(lambda: ((Q @ D.T) >= T[:, None]).nonzero())
    def deft():
        P = svde.probe_vectors(docs)
        m = (Q @ D.T) >= T[:, None]
        S = K @ P.half().T
        return (m & (S >= 0)).nonzero()
    r["deft_ms"] = timeit(deft)
    npairs = int(((Q @ D.T) >= T[:, None]).sum())
    r["reach_pairs"] = npairs
    subq = [O.trace(c, v)[h]["subq"] for c, v, h in hops][:64]; ans = [O.trace(c, v)[h]["ans"] for c, v, h in hops][:64]
    t = timeit(lambda: nli((docs * 64)[:64], [f"The answer to the question '{q}' is {a}." for q, a in zip(subq, ans)]), 2)
    r["nli_ms"] = t / 64 * npairs
    if cross is not None:
        with torch.no_grad():
            t = timeit(lambda: cross.cross_logits(subq, ans, (docs * 4)[:64]), 2)
        r["cross_ms"] = t / 64 * npairs
    print(r, flush=True)
    res["scal"].append(r)
    jdump(res, f"{OUT}/bench_{args.wl}.json")

if args.latency:
    from rag import RAG
    llm = LLMWrap(mem=0.45)
    enc = Encoder("cuda:0")
    index = DenseIndex(cap=len(O.texts) + 10)
    alive = O.alive[0].cpu().numpy()
    ids = [i for i in range(len(O.texts)) if alive[i]]
    index.add([O.doc_of_row[i] for i in ids], [O.texts[i] for i in ids], O.E[ids].half())
    rag = RAG(llm, enc, index, k=O.k)
    lat = []
    for c in O.cids[:60]:
        t = time.time(); rag.run([O.case[c]["q"]]); lat.append(time.time() - t)
    hit = []
    for c in O.cids[:200]:
        t = time.time(); e = enc.encode([O.case[c]["q"]], query=True); torch.cuda.synchronize(); hit.append(time.time() - t)
    ver = []
    for c in O.cids[:200]:
        t = time.time()
        for hop in O.trace(c, 0):
            s = O.qemb[hop["subq"]] @ O.E.T; torch.topk(s, O.k)
        torch.cuda.synchronize(); ver.append(time.time() - t + hit[0])
    res["lat"] = {"miss_ms_p50": 1000 * np.median(lat), "miss_ms_p95": 1000 * np.percentile(lat, 95), "miss_ms_mean": 1000 * np.mean(lat),
                  "hit_ms_p50": 1000 * np.median(hit), "verify_ms_p50": 1000 * np.median(ver), "verify_ms_p95": 1000 * np.percentile(ver, 95)}
    print(res["lat"], flush=True)
    jdump(res, f"{OUT}/bench_{args.wl}.json")
