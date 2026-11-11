"""End-to-end serving of the deployed read path on a dedicated GPU: one INT4 reader, one asynchronous vLLM engine with
prefix caching, the margin-transport reader on the same GPU, Poisson request arrivals, update batches in between.

Request logs come from run_closed2.py --dump (same cache decisions as the closed loop):
  exact  : every request to an entry with unverified hops certifies it (hop-local, hop states shared across entries)
  deft   : request-randomised certification (alpha 0.01, rho_min 0.05); the margin-transport reader scores an entry's
           cumulative edit the first time it is requested after a touch, inside that request
Certification modes: regen (re-generate each re-read hop) or spec (two-phase speculative verification of the cached
tokens: one prefill of the prompt, then the answer-position suffixes concurrently through the prefix cache; no
whole-prompt log-probabilities).  A rejected or changed hop resumes decoding and recomputes the suffix.
Reports throughput and p50/p95/p99 latency of all requests and of requests that needed the reader, reader and
scorer work.  Output: runs/<wl>/serve2/results.json
"""
import argparse, asyncio, pickle, time, random, uuid, collections
import numpy as np
from common import *
from simulate import Oracle
from mcert import MarginBook
from rag import HOP, DECOMP
from deltar import load_dr, score_examples

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--rates", default="200,400")
ap.add_argument("--scorer", default="mtr:_s0")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/serve2"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")
MB = MarginBook(args.wl, "test")
LOG = pickle.load(open(f"{RUNS}/{args.wl}/closed2/reqlog.pkl", "rb"))
v_, tag_ = args.scorer.split(":")
MODEL = load_dr(v_, tag_)

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
eng = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=LLM_PATH, quantization="awq_marlin", gpu_memory_utilization=float(os.environ.get("LLM_MEM", 0.7)),
    max_model_len=4096, enable_prefix_caching=True, seed=0, max_num_seqs=int(os.environ.get("LLM_SEQS", 64)),
    disable_log_requests=True))
tok = None
GEN_H = SamplingParams(temperature=0, max_tokens=32)
GEN_D = SamplingParams(temperature=0, max_tokens=48)
ONE = SamplingParams(temperature=0, max_tokens=1, logprobs=2)
CALLS = collections.Counter()
SCORE_MS = []


def chat_ids(u):
    s = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def hop_ids(q, rows):
    return chat_ids(HOP.format(docs="\n".join(f"[{i + 1}] {O.texts[r]}" for i, r in enumerate(rows)), q=q))


def decomp_ids(q, hist):
    h = "\n".join(f"({j + 1}) {a} -> {b}" for j, (a, b) in enumerate(hist)) or "(nothing yet)"
    return chat_ids(DECOMP.format(q=q, hist=h))


async def call(ids, sp, kind):
    CALLS[kind] += 1
    async for out in eng.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=uuid.uuid4().hex):
        pass
    return out


async def verify(ids, draft):
    """two-phase verification: prefill the prompt once, then all answer-position suffixes concurrently"""
    CALLS["verify"] += 1
    await call(ids, ONE, "verify_p1")
    if len(draft) > 1:
        await asyncio.gather(*[call(ids + draft[:i], ONE, "verify_p2") for i in range(1, len(draft))])


async def stage_recompute(c, v, start):
    tr = O.trace(c, v); q = O.case[c]["q"]
    for j in range(start, len(tr)):
        await call(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr[:j]]), GEN_D, "decomp")
        await call(hop_ids(tr[j]["subq"], tr[j]["rows"]), GEN_H, "reader")
    await call(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr]), GEN_D, "decomp")


async def certify(c, vo, vn, spec, shared):
    to, tn = O.trace(c, vo), O.trace(c, vn)
    hops, start, changed = [], None, None
    for j, (ho, hn) in enumerate(zip(to, tn)):
        if ho["subq"] != hn["subq"]:
            start = j; break                              # the decomposition differs: recompute from hop j
        if ho["rows"] != hn["rows"]:
            key = (ho["subq"], tuple(hn["rows"]), ho["ans"])
            if key not in shared:                         # hop states verified once, shared across entries
                hops.append(j); shared.add(key)
            if ho["ans"] != hn["ans"]:
                changed, start = j, j + 1; break          # the hop's new answer is known after its re-read
    if start is None and (len(to) != len(tn) or O.ans(c, vo) != O.ans(c, vn)):
        start = min(len(to), len(tn))
    if spec:
        await asyncio.gather(*[verify(hop_ids(tn[j]["subq"], tn[j]["rows"]), MB.tokens(to[j]["subq"], to[j]["rows"]) or [tok.eos_token_id]) for j in hops])
        if changed in hops:                               # a rejection resumes decoding after the verified prefix
            await call(hop_ids(tn[changed]["subq"], tn[changed]["rows"]), GEN_H, "resume")
    else:
        for j in hops:
            await call(hop_ids(tn[j]["subq"], tn[j]["rows"]), GEN_H, "reader")
    if start is not None:
        await stage_recompute(c, vn, start)


def score(items):
    t = time.time()
    exs = [{"subq": q, "ans": a, "K": [(0, x) for x in K], "K2": [(0, x) for x in K2], "mu": m} for q, a, K, K2, m in items]
    score_examples(MODEL, exs, {}, lambda d: O.texts[d[1]], bs=32)
    torch.cuda.synchronize()
    SCORE_MS.append(1000 * (time.time() - t))


async def serve(policy, spec, rate):
    log = [x for x in LOG[policy] if x[0] <= args.batches]
    gaps = np.random.default_rng(0).exponential(1.0 / rate, len(log))
    lat, lat_llm, tasks = [], [], []
    CALLS.clear(); SCORE_MS.clear()
    shared = set()
    loop = asyncio.get_event_loop()
    t0 = time.time(); t = t0
    for i, x in enumerate(log):
        b, kind, c, vo, vn, items = x
        t += gaps[i]
        await asyncio.sleep(max(0.0, t - time.time()))
        if kind == "hit" and not items:
            lat.append(0.0); continue

        async def one(kind=kind, c=c, vo=vo, vn=vn, items=items, ts=time.time()):
            if items:
                await loop.run_in_executor(None, score, items)
            if kind == "cert":
                await certify(c, vo, vn, spec, shared)
                d = time.time() - ts; lat.append(d); lat_llm.append(d)
            else:
                lat.append(time.time() - ts)
        tasks.append(asyncio.ensure_future(one()))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    L, LL = np.array(lat), (np.array(lat_llm) if lat_llm else np.zeros(1))
    return {"requests": len(log), "reader_requests": len(lat_llm), "wall_s": wall, "throughput_rps": len(log) / wall,
            "p50_ms": 1000 * float(np.percentile(L, 50)), "p95_ms": 1000 * float(np.percentile(L, 95)), "p99_ms": 1000 * float(np.percentile(L, 99)),
            "mean_ms": 1000 * float(L.mean()),
            "llm_p50_ms": 1000 * float(np.percentile(LL, 50)), "llm_p95_ms": 1000 * float(np.percentile(LL, 95)), "llm_p99_ms": 1000 * float(np.percentile(LL, 99)),
            "scored_requests": len(SCORE_MS), "score_ms_p50": float(np.percentile(SCORE_MS, 50)) if SCORE_MS else 0.0,
            "score_ms_p95": float(np.percentile(SCORE_MS, 95)) if SCORE_MS else 0.0, "calls": dict(CALLS)}


async def main():
    global tok
    tok = await eng.get_tokenizer()
    await call(chat_ids("warm up"), GEN_H, "warm")
    score([(O.trace(O.cids[0], 0)[0]["subq"], "x", tuple(O.trace(O.cids[0], 0)[0]["rows"]), tuple(O.trace(O.cids[0], 0)[0]["rows"]), 1.0)])
    R = {"meta": {"wl": args.wl, "batches": args.batches, "scorer": args.scorer}}
    for rate in [float(r) for r in args.rates.split(",")]:
        for policy, spec in (("exact", False), ("exact", True), ("deft", True), ("deft", False)):
            k = f"{policy}+{'spec' if spec else 'regen'}@{rate:g}"
            R[k] = await serve(policy, spec, rate)
            print(k, {a: (round(b_, 1) if isinstance(b_, float) else b_) for a, b_ in R[k].items()}, flush=True)
            jdump(R, f"{OUT}/results.json")
    print("done", flush=True)

asyncio.get_event_loop().run_until_complete(main())
