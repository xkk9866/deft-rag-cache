"""End-to-end serving of the deployed read path with the speculator: one INT4 reader in an asynchronous vLLM engine
(prefix caching), the speculator (a small same-family reader, HF, bf16) on the same GPU, Poisson request arrivals,
update batches in between.  Request logs come from run_closed2.py --dump <scorer> (same cache decisions as the closed
loop):  exact = every request to an entry with unverified hops certifies it;  deft = request-randomised certification
(alpha 0.01, rho_min 0.05), the speculator measures an entry's unverified hops the first time it is requested after a
touch, inside that request.  Certification re-generates each re-read hop (one request with top-20 log-probabilities)
and recomputes the suffix after the first changed hop; reader calls on hop states already in the shared hop-state
table (verified or computed by any entry) are skipped (shared computation).
Output: runs/<wl>/serve3/results.json"""
import argparse, asyncio, pickle, time, uuid, collections
import numpy as np
from common import *
from simulate import Oracle
from mcert import MarginBook
from rag import HOP, DECOMP

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--rates", default="200,400")
ap.add_argument("--scorer", default="spec:q06")
ap.add_argument("--log", default="", help="request log file (default closed2/reqlog_<scorer>.pkl)")
ap.add_argument("--policies", default="exact,deft")
args = ap.parse_args()
OUT = f"{RUNS}/{args.wl}/serve3"; os.makedirs(OUT, exist_ok=True)
O = Oracle(args.wl, "test")
MB = MarginBook(args.wl, "test")
lf = args.log or f"{RUNS}/{args.wl}/closed2/reqlog_{args.scorer.replace(':', '_')}.pkl"
LOG = pickle.load(open(lf, "rb"))
parts = args.scorer.split(":")
if parts[0] == "spec":
    from spec import Speculator
    SPEC = Speculator(parts[1], parts[2] if len(parts) > 2 and parts[2] not in ("zs", "did", "raw") else "")
    MODEL = None
else:
    from deltar import load_dr, score_examples
    MODEL = load_dr(parts[0], parts[1]); SPEC = None

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from vllm.inputs import TokensPrompt


def start_engine(tries=int(os.environ.get("ENGINE_TRIES", 1)), wait=45):
    """on a GPU shared with jobs that start at any moment, the engine may lose the race for memory while it loads;
    retry in place (the speculator stays resident), re-deriving the memory fraction from the current free memory"""
    need = float(os.environ.get("LLM_NEED_MB", 7500)) * 2 ** 20
    for t in range(tries):
        free, total = torch.cuda.mem_get_info()
        frac = min(0.95, (total - free + need) / total)
        try:
            return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
                model=LLM_PATH, quantization="awq_marlin", gpu_memory_utilization=frac,
                max_model_len=int(os.environ.get("LLM_MAXLEN", 4096)), enable_prefix_caching=True, seed=0,
                max_num_seqs=int(os.environ.get("LLM_SEQS", 64)), enforce_eager=bool(os.environ.get("LLM_EAGER")),
                **({"max_num_batched_tokens": int(os.environ["LLM_BTOK"])} if os.environ.get("LLM_BTOK") else {}),
                disable_log_requests=True))
        except Exception as e:                          # OOM or "no memory for cache blocks": someone else got there first
            print(f"engine start failed (try {t + 1}/{tries}, frac {frac:.3f}, free {free / 2**30:.1f} GiB): {repr(e)[:160]}", flush=True)
            torch.cuda.empty_cache(); time.sleep(wait)
    raise SystemExit("engine could not be started")


eng = start_engine()
tok = None
GEN_H = SamplingParams(temperature=0, max_tokens=32, logprobs=20)
GEN_D = SamplingParams(temperature=0, max_tokens=48)
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


async def read_hop(subq, rows, shared):
    key = (subq, tuple(rows))
    if key in shared:
        CALLS["shared_hit"] += 1; return
    shared.add(key)
    await call(hop_ids(subq, rows), GEN_H, "reader")


async def stage_recompute(c, v, start, shared):
    tr = O.trace(c, v); q = O.case[c]["q"]
    for j in range(start, len(tr)):
        await call(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr[:j]]), GEN_D, "decomp")
        await read_hop(tr[j]["subq"], tr[j]["rows"], shared)
    await call(decomp_ids(q, [(h["subq"], h["ans"]) for h in tr]), GEN_D, "decomp")


async def certify(c, vo, vn, shared):
    to, tn = O.trace(c, vo), O.trace(c, vn)
    hops, start = [], None
    for j, (ho, hn) in enumerate(zip(to, tn)):
        if ho["subq"] != hn["subq"]:
            start = j; break
        if ho["rows"] != hn["rows"]:
            hops.append(j)
            if ho["ans"] != hn["ans"]:
                start = j + 1; break
    if start is None and (len(to) != len(tn) or O.ans(c, vo) != O.ans(c, vn)):
        start = min(len(to), len(tn))
    await asyncio.gather(*[read_hop(tn[j]["subq"], tn[j]["rows"], shared) for j in hops])
    if start is not None:
        await stage_recompute(c, vn, start, shared)


def score(items):
    if SPEC is not None:
        SPEC.margins(O.texts, [(q, K2, tuple(MB.tokens(q, K) or ())) for q, a, K, K2, m in items], bs=32)
    else:
        exs = [{"subq": q, "ans": a, "K": [(0, x) for x in K], "K2": [(0, x) for x in K2], "mu": m} for q, a, K, K2, m in items]
        score_examples(MODEL, exs, {}, lambda d: O.texts[d[1]], bs=32)
    torch.cuda.synchronize()


class MicroBatcher:
    """requests that arrive while unverified hand their hop states to one scoring loop, which runs a single forward
    pass over everything pending (at most every WINDOW seconds); the speculator never competes with itself"""
    WINDOW = 0.01

    def __init__(self):
        self.pending, self.ev, self.task = [], None, None

    async def submit(self, items):
        loop = asyncio.get_event_loop()
        fut = loop.create_future(); t0 = time.time()
        self.pending.append((items, fut))
        if self.task is None or self.task.done():
            self.task = asyncio.ensure_future(self.run())
        await fut
        SCORE_MS.append(1000 * (time.time() - t0))

    async def run(self):
        loop = asyncio.get_event_loop()
        while self.pending:
            await asyncio.sleep(self.WINDOW)
            batch, self.pending = self.pending, []
            flat = list(dict.fromkeys(it for items, _ in batch for it in items))
            CALLS["spec_batches"] += 1; CALLS["spec_items"] += len(flat)
            try:
                await loop.run_in_executor(None, score, flat)
            finally:
                for _, fut in batch:
                    if not fut.done():
                        fut.set_result(True)


BATCHER = MicroBatcher()


async def serve(policy, rate):
    log = [x for x in LOG[policy] if x[0] <= args.batches]
    gaps = np.random.default_rng(0).exponential(1.0 / rate, len(log))
    lat, lat_llm, tasks = [], [], []
    CALLS.clear(); SCORE_MS.clear()
    if SPEC is not None:
        SPEC.memo.clear(); SPEC.pmemo.clear()
    shared = set((hop["subq"], tuple(hop["rows"])) for c in O.cids for hop in O.trace(c, 0))
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
                await BATCHER.submit(items)
            if kind == "cert":
                await certify(c, vo, vn, shared)
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
    h0 = O.trace(O.cids[0], 0)[0]
    score([(h0["subq"], h0["ans"], tuple(h0["rows"]), tuple(h0["rows"]), 1.0)])
    dst = f"{OUT}/results.json"
    R = jload(dst) if os.path.exists(dst) else {}
    R["meta"] = {"wl": args.wl, "batches": args.batches, "scorer": args.scorer, "log": lf}
    for rate in [float(r) for r in args.rates.split(",")]:
        for policy in args.policies.split(","):
            k = f"{policy}+regen@{rate:g}"
            R[k] = await serve(policy, rate)
            print(k, {a: (round(b_, 1) if isinstance(b_, float) else b_) for a, b_ in R[k].items()}, flush=True)
            jdump(R, f"{OUT}/results.json")
    print("done", flush=True)

asyncio.get_event_loop().run_until_complete(main())
