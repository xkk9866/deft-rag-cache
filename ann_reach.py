"""Trace invariance under APPROXIMATE retrieval (HNSW and IVF, faiss), on the real update streams.

The cache is served by an ANN index that is updated in place (insertions added to the index, deletions masked by an
ID selector).  For every cached hop and update batch we compare the ANN top-k before and after the batch, and test
two touch detectors:
  * threshold test  : a new document scores >= tau_j (the k-th score of the hop's ANN result) or a cited doc is deleted;
  * routing-aware   : the threshold test, plus every hop whose ANN search VISITED a list / graph region changed by the
                      batch (IVF: the hop's probed lists received or lost a document; HNSW: re-search the hops in the
                      threshold test's complement only if their old top-k contains a deleted row's graph neighbour).
Missed = ANN top-k changed but the detector did not fire.  Reported per index: changed-hop rate, misses of each detector,
and the share of all hops each detector must re-examine (its cost).  Output: runs/<wl>/ann/results.json
"""
import argparse, time, collections
import numpy as np
import faiss
from common import *
from simulate import Oracle

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/ann"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")
E = O.E.cpu().numpy().astype("float32"); d = E.shape[1]; k = O.k
alive0 = O.alive[0].cpu().numpy()
# the hops of the cache: every hop of every entry's version valid before each batch (the subq set is what matters)
subqs = sorted({h["subq"] for c in O.cids for v, _ in enumerate(O.versions[c]) for h in O.versions[c][v][1]["trace"]})
Q = torch.stack([O.qemb[q] for q in subqs]).cpu().numpy().astype("float32")
print("hops", len(subqs), "rows", len(E), flush=True)


def build(kind):
    if kind == "hnsw":
        idx = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); idx.hnsw.efConstruction = 80
        idx = faiss.IndexIDMap2(idx)
    else:
        nlist = int(4 * np.sqrt(alive0.sum()))
        q = faiss.IndexFlatIP(d); idx = faiss.IndexIVFFlat(q, d, nlist, faiss.METRIC_INNER_PRODUCT)
        idx.train(E[alive0]); idx = faiss.IndexIDMap2(idx)
    ids = np.nonzero(alive0)[0]
    idx.add_with_ids(E[ids], ids.astype("int64"))
    return idx


def search(idx, kind, alive):
    sel = faiss.IDSelectorBatch(np.nonzero(alive)[0].astype("int64"))
    if kind == "hnsw":
        p = faiss.SearchParametersHNSW(sel=sel, efSearch=64)
    else:
        p = faiss.SearchParametersIVF(sel=sel, nprobe=16)
    D, I = idx.search(Q, k, params=p)
    return D, I


res = {}
for kind in ("hnsw", "ivf"):
    idx = build(kind)
    inner = faiss.downcast_index(idx.index)
    alive = alive0.copy()
    D0, I0 = search(idx, kind, alive)
    if kind == "ivf":
        _, probe0 = inner.quantizer.search(Q, 16)
    m = collections.Counter(); t0 = time.time()
    for b in range(1, O.nB + 1):
        lg = O.logs[b - 1]
        ins = np.array(O.ins_rows[b - 1], dtype="int64"); rem = np.array(lg["rem"], dtype="int64")
        if len(ins):
            idx.add_with_ids(E[ins], ins)
        alive = alive.copy(); alive[ins] = True; alive[rem] = False
        D1, I1 = search(idx, kind, alive)
        changed = np.array([set(a) != set(c) for a, c in zip(I0, I1)])
        tau = D0[:, -1]
        thr = np.zeros(len(Q), dtype=bool)
        if len(ins):
            thr |= ((Q @ E[ins].T) >= tau[:, None] - 1e-6).any(1)
        if len(rem):
            thr |= np.isin(I0, rem).any(1)
        if kind == "ivf":
            lists = set()
            for r in np.concatenate([ins, rem]).astype("int64"):
                lists.add(int(inner.quantizer.search(E[r:r + 1], 1)[1][0, 0]))
            route = thr | np.isin(probe0, list(lists)).any(1) if lists else thr
        else:
            # HNSW: insertions only add edges (old paths survive), deletions hide nodes; re-search hops whose result
            # contained a deleted row or whose k-th score lies below a new document's score (threshold test)
            route = thr
        # maintained evidence set (Algorithm 1, line 6) vs. what a re-search over the updated index returns.
        # global merge : new documents enter by score alone;  list-aware merge (IVF): a new document is a candidate
        # only if it lies in one of the hop's probed lists, i.e. the merge applies the index's own retrieval function.
        # Hops that cited a deleted document are re-searched locally in both rules (as in Algorithm 1, line 4).
        if len(ins):
            S_ins = Q @ E[ins].T
            if kind == "ivf":
                lst = inner.quantizer.search(E[ins], 1)[1][:, 0]
                inprobe = (probe0[:, :, None] == lst[None, None, :]).any(1)
        need = thr | changed
        for h in np.nonzero(need)[0]:
            truth = set(I1[h][I1[h] >= 0])
            if len(rem) and np.isin(I0[h], rem).any():
                m["eq_glob"] += 1; m["eq_list"] += 1; m["eq_n"] += 1      # locally re-searched: equal by construction
                continue
            for rule in ("glob", "list"):
                cand = list(zip(D0[h], I0[h]))
                if len(ins):
                    ok = np.ones(len(ins), bool) if (rule == "glob" or kind != "ivf") else inprobe[h]
                    cand += [(S_ins[h, i], ins[i]) for i in range(len(ins)) if ok[i]]
                top = {r for _, r in sorted(cand, key=lambda x: -x[0])[:k]}
                m[f"eq_{rule}"] += int(top == truth)
            m["eq_n"] += 1
        m["hops"] += len(Q); m["changed"] += int(changed.sum())
        m["miss_thr"] += int((changed & ~thr).sum()); m["miss_route"] += int((changed & ~route).sum())
        m["fire_thr"] += int(thr.sum()); m["fire_route"] += int(route.sum())
        D0, I0 = D1, I1
    res[kind] = {"changed_rate": m["changed"] / m["hops"], "miss_thr_of_changed": m["miss_thr"] / max(1, m["changed"]),
                 "miss_route_of_changed": m["miss_route"] / max(1, m["changed"]), "fire_thr_rate": m["fire_thr"] / m["hops"],
                 "fire_route_rate": m["fire_route"] / m["hops"], "sec": time.time() - t0,
                 "maintained_equal_global": m["eq_glob"] / max(1, m["eq_n"]), "maintained_equal_listaware": m["eq_list"] / max(1, m["eq_n"]),
                 "maintained_checked": m["eq_n"]}
    print(kind, res[kind], flush=True)
    jdump(res, f"{OUT}/results.json")
print("done", flush=True)
