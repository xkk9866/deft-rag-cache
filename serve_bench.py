"""End-to-end serving under concurrency: one INT4 reader on one GPU serves the request stream of the closed-loop
replay (run_closed.py --dump) while update batches arrive; every LLM call goes through the same async vLLM engine.

Policies (same request stream, same cache decisions as in the closed loop):
  exact+regen  : trace-exact maintenance, suspects certified by re-generating each changed hop in trace order
  exact+spec   : trace-exact maintenance, suspects certified by speculative verification (all changed hops at once)
  deftx+regen  : DEFT-X filter (alpha 0.1, audited adaptive control, pi 0.25), re-generation certification
  deftx+spec   : DEFT-X filter, speculative certification
Work per request: hit -> none; miss -> full traced recomputation (2m+1 sequential calls); certification -> hop
re-reads of the hops whose evidence changed since the trusted trace, then recomputation of the suffix from the first
changed hop if the answer changed.  Audits are write-path work injected when their batch arrives.
Requests arrive as a Poisson process at rate `--rate`; update batch b arrives before its requests.
Reports throughput, p50/p95/p99 latency of all requests and of requests that needed the reader, GPU busy seconds,
and reader-call counts.  Output: runs/<wl>/serve/results.json
"""
import argparse, asyncio, pickle, time, random, uuid, collections
import numpy as np
os_env = __import__("os").environ
os_env.setdefault("VLLM_USE_V1", "0")
from common import *
from simulate import Oracle
from rag import HOP, DECOMP

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--rates", default="200,400,800")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/serve"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")
LOG = pickle.load(open(f"{RUNS}/{args.wl}/closed/reqlog.pkl", "rb"))

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt
eng = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
    model=LLM_PATH, quantization="awq_marlin", gpu_memory_utilization=float(os_env.get("LLM_MEM", 0.6)),
    max_model_len=int(os_env.get("LLM_MAXLEN", 4096)), enable_prefix_caching=False, seed=0,
    max_num_seqs=int(os_env.get("LLM_SEQS", 32)), enforce_eager=bool(os_env.get("LLM_EAGER")), disable_log_requests=True,
    enable_chunked_prefill=False, max_num_batched_tokens=int(os_env.get("LLM_MAXLEN", 4096))))   # bounds the prompt-logprob tensor
tok = None


def chat_ids(u):
    s = tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(s, add_special_tokens=False)["input_ids"]


def hop_ids(q, rows):
    return chat_ids(HOP.format(docs="\n".join(f"[{i + 1}] {O.texts[r]}" for i, r in enumerate(rows)), q=q))


def decomp_ids(q, hist):
    h = "\n".join(f"({j + 1}) {a} -> {b}" for j, (a, b) in enumerate(hist)) or "(nothing yet)"
    return chat_ids(DECOMP.format(q=q, hist=h))


GEN_H = SamplingParams(temperature=0, max_tokens=32)
GEN_D = SamplingParams(temperature=0, max_tokens=48)
VER = SamplingParams(temperature=0, max_tokens=1)   # one prefill over (prompt, cached answer); verdict correctness is measured by spec_cert.py
CALLS = collections.Counter()


async def call(ids, sp, kind):
    CALLS[kind] += 1
    async for out in eng.generate(TokensPrompt(prompt_token_ids=ids), sp, request_id=uuid.uuid4().hex):
        pass
    return out


def recompute_plan(c, v, start=0):
    """sequential stages of a traced recomputation of version v from hop `start` (decomposer + reader per hop, final)"""
    tr = O.trace(c, v); q = O.case[c]["q"]; st = []
    for j in range(start, len(tr)):
        st.append([(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr[:j]]), GEN_D, "decomp")])
        st.append([(hop_ids(tr[j]["subq"], tr[j]["rows"]), GEN_H, "reader")])
    st.append([(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr]), GEN_D, "decomp")])
    return st


def cert_plan(c, vo, vn, spec):
    """hop-local certification from the trusted version vo to the current version vn"""
    to, tn = O.trace(c, vo), O.trace(c, vn)
    hops, first_change = [], None
    for j, (ho, hn) in enumerate(zip(to, tn)):
        if ho["subq"] != hn["subq"]:
            first_change = j if first_change is None else first_change; break
        if ho["rows"] != hn["rows"]:
            hops.append(j)
            if ho["ans"] != hn["ans"]:
                first_change = j; break
    if first_change is None and (len(to) != len(tn) or O.ans(c, vo) != O.ans(c, vn)):
        first_change = min(len(to), len(tn))
    if spec:
        draft = {j: tok(to[j]["ans"], add_special_tokens=False)["input_ids"] + [tok.convert_tokens_to_ids("<|im_end|>")] for j in hops}
        st = [[(hop_ids(tn[j]["subq"], tn[j]["rows"]) + draft[j], VER, "verify") for j in hops]] if hops else []
    else:
        st = [[(hop_ids(tn[j]["subq"], tn[j]["rows"]), GEN_H, "reader")] for j in hops]
    if first_change is not None:
        # the hop that changed was already re-read (its new answer is known): resume with the next decomposition
        st += recompute_plan(c, vn, first_change)[(2 if first_change in hops else 0):]
    return st


async def run_plan(plan):
    for stage in plan:
        await asyncio.gather(*[call(ids, sp, k) for ids, sp, k in stage])


async def serve(policy, spec, rate):
    log = [x for x in LOG[policy] if x[0] <= args.batches]
    rng = random.Random(0)
    reqs = [x for x in log if x[1] != "audit"]
    gaps = np.random.default_rng(0).exponential(1.0 / rate, len(reqs))
    lat, lat_llm, tasks = [], [], []
    CALLS.clear()
    t0 = time.time(); t = t0; ri = 0
    by_b = collections.defaultdict(list)
    for x in log:
        by_b[x[0]].append(x)
    for b in sorted(by_b):
        for x in by_b[b]:
            if x[1] == "audit":                               # write-path work of an audited batch
                tasks.append(asyncio.ensure_future(run_plan(cert_plan(x[2], x[3], x[4], spec))))
        for x in by_b[b]:
            if x[1] == "audit":
                continue
            t += gaps[ri]; ri += 1
            await asyncio.sleep(max(0.0, t - time.time()))
            if x[1] in ("hit",):
                lat.append(0.0); continue
            plan = recompute_plan(x[2], x[4]) if x[1] == "miss" else cert_plan(x[2], x[3], x[4], spec)

            async def one(plan=plan, ts=time.time()):
                await run_plan(plan); d = time.time() - ts; lat.append(d); lat_llm.append(d)
            tasks.append(asyncio.ensure_future(one()))
    await asyncio.gather(*tasks)
    wall = time.time() - t0
    L, LL = np.array(lat), np.array(lat_llm) if lat_llm else np.zeros(1)
    return {"requests": len(reqs), "llm_requests": len(lat_llm), "wall_s": wall, "throughput_rps": len(reqs) / wall,
            "p50_ms": 1000 * float(np.percentile(L, 50)), "p95_ms": 1000 * float(np.percentile(L, 95)), "p99_ms": 1000 * float(np.percentile(L, 99)),
            "llm_p50_ms": 1000 * float(np.percentile(LL, 50)), "llm_p95_ms": 1000 * float(np.percentile(LL, 95)), "llm_p99_ms": 1000 * float(np.percentile(LL, 99)),
            "calls": dict(CALLS)}


async def main():
    global tok
    tok = await eng.get_tokenizer()
    await call(chat_ids("warm up"), GEN_H, "warm")
    R = {"meta": {"wl": args.wl, "batches": args.batches}}
    for rate in [float(r) for r in args.rates.split(",")]:
        for policy in ("exact", "deftx"):
            for spec in (False, True):
                k = f"{policy}+{'spec' if spec else 'regen'}@{rate:g}"
                R[k] = await serve(policy, spec, rate)
                print(k, {a: (round(b, 1) if isinstance(b, float) else b) for a, b in R[k].items()}, flush=True)
                jdump(R, f"{OUT}/results.json")
    print("done", flush=True)

asyncio.get_event_loop().run_until_complete(main())
