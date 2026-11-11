"""Ground-truth replay of an update stream.

Maintains every cached entry exactly fresh (recompute-consistent) with the trace-exact rule
(Thm. 1: only entries whose recorded trace is reached by an inserted row, or that cite a removed row,
can change).  For every batch it logs which entries were reachable, by which (hop, row) pairs,
and whether their final answer actually changed.  These logs are the oracle for all policies.
"""
import argparse, random, time
import numpy as np
from common import *
from rag import RAG

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--split", required=True)
ap.add_argument("--B", type=int, default=20)
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--max_cases", type=int, default=100000)
ap.add_argument("--noise_every", type=int, default=5)
args = ap.parse_args()
if args.split == "train" and len(jload(f"{DATA}/{args.wl}/split_train.json")["events"]) > 2000:
    args.B = 40          # long training streams are replayed in coarser batches (labels only)
torch.manual_seed(0); random.seed(0)
OUT = f"{RUNS}/{args.wl}/{args.split}"
os.makedirs(OUT, exist_ok=True)

corpus = jload(f"{DATA}/{args.wl}/corpus.json")
sp = jload(f"{DATA}/{args.wl}/split_{args.split}.json")
cases = sp["cases"][: args.max_cases]
events = sp["events"]
print(args, "cases", len(cases), "events", len(events), "corpus", len(corpus), flush=True)

enc = Encoder("cuda:0")
llm = LLMWrap(mem=0.55)
index = DenseIndex(cap=len(corpus) + len(events) + 100)
ids = sorted(corpus)
t = time.time()
E = enc.encode([corpus[i] for i in ids])
index.add(ids, [corpus[i] for i in ids], E)
print("indexed", len(ids), f"{time.time()-t:.1f}s", flush=True)
rag = RAG(llm, enc, index, k=args.k)
qtext = {c["cid"]: c["q"] for c in cases}

t = time.time()
res = rag.run([c["q"] for c in cases])
t_init = time.time() - t
entries = {c["cid"]: r for c, r in zip(cases, res)}
versions = {cid: [(0, r)] for cid, r in entries.items()}
calls0 = llm.calls
print(f"initial answers {len(cases)} in {t_init:.1f}s, llm calls {calls0}", flush=True)
acc = np.mean([em_alias(r["answer"], c["answer_alias"]) or contains_alias(r["answer"], c["answer_alias"]) for c, r in zip(cases, res)])
print("initial accuracy vs pre-update gold (contains):", acc, flush=True)


def hop_keys(ents):
    """stack [u(q_h); -tau_h] for every hop of every entry."""
    owners, K = [], []
    for cid, r in ents.items():
        for h, hop in enumerate(r["trace"]):
            owners.append((cid, h))
            K.append(torch.cat([rag.qcache[hop["subq"]].float(), torch.tensor([-hop["tau"]], device="cuda:0")]))
    return owners, (torch.stack(K) if K else None)


logs = []
batches = [events[i:i + args.B] for i in range(0, len(events), args.B)]
pipeline_time, recomputed = 0.0, 0
noise = []
for b, batch in enumerate(batches, start=1):
    ins_rows, rem_rows, ins_info = [], [], []
    texts = [e["text"] for e in batch if e["op"] in ("insert", "modify")]
    embs = enc.encode(texts) if texts else None
    j = 0
    for e in batch:
        if e["op"] == "delete":
            r = index.delete(e["doc"])
            if r is not None:
                rem_rows.append(r)
        else:
            old = index.row_of_doc.get(e["doc"])
            if e["op"] == "modify" and old is not None:
                rem_rows.append(old)
            rows = index.add([e["doc"]], [e["text"]], embs[j:j + 1]); j += 1
            ins_rows += rows
            ins_info.append((rows[0], e["kind"], e["doc"]))
    # ---- trace-exact reachability
    owners, K = hop_keys(entries)
    reach = {}
    if ins_rows and K is not None:
        P = torch.cat([index.E[ins_rows].float(), torch.ones(len(ins_rows), 1, device="cuda:0")], 1)
        M = (K @ P.T) >= -1e-4
        for oi, ri in M.nonzero().tolist():
            cid, h = owners[oi]
            reach.setdefault(cid, {"ins": [], "rem": False})["ins"].append((h, ins_rows[ri]))
    rs = set(rem_rows)
    if rs:
        for cid, r in entries.items():
            if any(x in rs for hop in r["trace"] for x in hop["rows"]):
                reach.setdefault(cid, {"ins": [], "rem": False})["rem"] = True
    # ---- recompute reachable entries
    rc = sorted(reach)
    t = time.time()
    newres = rag.run([qtext[cid] for cid in rc]) if rc else []
    pipeline_time += time.time() - t; recomputed += len(rc)
    changed = {}
    for cid, r in zip(rc, newres):
        changed[cid] = not same_answer(r["answer"], entries[cid]["answer"])
        entries[cid] = r
        versions[cid].append((b, r))
    # ---- determinism check on unreachable entries
    if args.noise_every and b % args.noise_every == 0:
        pool = [cid for cid in entries if cid not in reach]
        smp = random.Random(b).sample(pool, min(48, len(pool)))
        chk = rag.run([qtext[cid] for cid in smp])
        noise.append((len(smp), sum(not same_answer(r["answer"], entries[cid]["answer"]) for cid, r in zip(smp, chk))))
    logs.append({"b": b, "ins": ins_info, "rem": rem_rows, "events": [(e["op"], e["kind"], e["doc"]) for e in batch],
                 "reach": reach, "changed": changed})
    nchg = sum(changed.values())
    print(f"batch {b}/{len(batches)} ins {len(ins_rows)} rem {len(rem_rows)} reach {len(rc)} changed {nchg} "
          f"calls {llm.calls}", flush=True)

final_acc_new = np.mean([contains_alias(entries[c["cid"]]["answer"], c["new_answer_alias"]) for c in cases])
print("noise (sampled unreachable recomputes, changed):", noise, flush=True)
print("final accuracy vs post-update gold (contains):", final_acc_new, flush=True)
qemb = {q: e.cpu().numpy() for q, e in rag.qcache.items()}
pdump({"cases": cases, "versions": versions, "logs": logs, "noise": noise, "texts": index.texts,
       "doc_of_row": index.doc_of_row, "E": index.E[:index.n].cpu().numpy(), "qemb": qemb,
       "init_rows": len(ids), "B": args.B, "k": args.k,
       "timing": {"init_s": t_init, "init_n": len(cases), "pipeline_s": pipeline_time, "recomputed": recomputed,
                  "llm_calls": llm.calls}}, f"{OUT}/oracle.pkl")
print("saved", OUT, flush=True)
