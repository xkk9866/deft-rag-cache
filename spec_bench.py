"""Per-hop cost of the defeat predictors on identical hop states: the speculator (same-family sibling, one teacher-forced
pass over prompt + cached tokens), the margin-transport cross-encoder (bge-reranker-v2-m3, 1024 tokens) and, if a
reader engine is available in memory terms, batched re-generation by the reader (vLLM).  All on the current GPU,
same states, same batch sizes.  Output: runs/<wl>/spec_bench.json"""
import argparse, time
import numpy as np
from common import *
from mcert import MarginBook
from spec import Speculator

ap = argparse.ArgumentParser()
ap.add_argument("--wl", default="wiki_t")
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--specs", default="q06,q17")
ap.add_argument("--adapters", default="", help="comma list aligned with --specs ('' = zero-shot)")
ap.add_argument("--mtr", default="mtr:_s0")
ap.add_argument("--reader", action="store_true", help="also time batched re-generation with the reader (needs vLLM memory)")
ap.add_argument("--bs", default="16,32")
ap.add_argument("--out", default="spec_bench")
args = ap.parse_args()

R = pload(f"{RUNS}/{args.wl}/test/oracle.pkl")
MB = MarginBook(args.wl, "test")
texts = R["texts"]
states = [(q, rows, tuple(st["tok"]), st["ans"]) for (q, rows), st in MB.S.items() if st["tok"]]
rng = np.random.default_rng(0); rng.shuffle(states)
states = states[: args.n]
dst = f"{RUNS}/{args.wl}/{args.out}.json"
out = jload(dst) if os.path.exists(dst) else {}
out.update({"wl": args.wl, "n": len(states), "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"), "session": time.strftime("%Y-%m-%d %H:%M")})
save = lambda: jdump(out, dst)                      # every measurement is persisted as soon as it exists
print(args.wl, "states", len(states), flush=True)

adapters = (args.adapters.split(",") + [""] * 8) if args.adapters else [""] * 8
for name, ad in zip([x for x in args.specs.split(",") if x], adapters):
    sp = Speculator(name, ad)
    for bs in [int(b) for b in args.bs.split(",")]:
        sp.memo.clear(); sp.time = 0.0; sp.calls = 0; sp.tokens = 0
        sp.margins(texts, [(q, rows, y) for q, rows, y, _ in states[:32]], bs=bs)      # warm-up
        sp.memo.clear(); sp.time = 0.0; sp.calls = 0; sp.tokens = 0
        torch.cuda.synchronize(); t = time.time()
        sp.margins(texts, [(q, rows, y) for q, rows, y, _ in states], bs=bs)
        torch.cuda.synchronize(); dt = time.time() - t
        out[f"spec_{name}{ad and '+' + ad}_bs{bs}_ms"] = 1000 * dt / len(states)
        out[f"spec_{name}_tokens"] = sp.tokens / len(states)
        print(name, ad, "bs", bs, f"{1000 * dt / len(states):.1f} ms/hop", f"{sp.tokens / len(states):.0f} tok/hop", flush=True)
    out[f"spec_{name}_params_M"] = sum(p.numel() for p in sp.lm.parameters()) / 1e6
    save()
    del sp; torch.cuda.empty_cache()

if args.mtr:
    from deltar import load_dr, score_examples
    v, tag = (args.mtr.split(":") + [""])[:2]
    M = load_dr(v, tag)
    exs = [{"subq": q, "ans": a, "K": [(0, x) for x in rows], "K2": [(0, x) for x in rows], "mu": MB.mu(q, rows)} for q, rows, y, a in states]
    for bs in [int(b) for b in args.bs.split(",")]:
        score_examples(M, exs[:32], {}, lambda d: texts[d[1]], bs=bs)
        torch.cuda.synchronize(); t = time.time()
        score_examples(M, exs, {}, lambda d: texts[d[1]], bs=bs)
        torch.cuda.synchronize(); dt = time.time() - t
        out[f"mtr_bs{bs}_ms"] = 1000 * dt / len(states)
        print("mtr bs", bs, f"{1000 * dt / len(states):.1f} ms/hop", flush=True)
    out["mtr_params_M"] = sum(p.numel() for p in M.parameters()) / 1e6
    save()
    del M; torch.cuda.empty_cache()

if args.reader:
    from rag import HOP
    llm = LLMWrap(mem=float(os.environ.get("LLM_MEM", 0.5)))
    tok = llm.tok
    def prompt(q, rows):
        docs = "\n".join(f"[{i + 1}] {texts[r]}" for i, r in enumerate(rows))
        return tok.apply_chat_template([{"role": "user", "content": HOP.format(docs=docs, q=q)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    SP = llm.SP(temperature=0, max_tokens=32, logprobs=20)
    ps = [prompt(q, rows) for q, rows, _, _ in states]
    llm.llm.generate(ps[:32], SP, use_tqdm=False)
    t = time.time(); llm.llm.generate(ps, SP, use_tqdm=False); dt = time.time() - t
    out["reader_regen_batched_ms"] = 1000 * dt / len(states)
    print("reader batched re-generation", f"{1000 * dt / len(states):.1f} ms/hop", flush=True)
    save()
    lat = []
    for p in ps[:64]:
        t = time.time(); llm.llm.generate([p], SP, use_tqdm=False); lat.append(1000 * (time.time() - t))
    out["reader_regen_single_p50_ms"] = float(np.percentile(lat, 50))
    save()
save()
print("BENCH DONE", out, flush=True)
