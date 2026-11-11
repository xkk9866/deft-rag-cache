"""Interleaved per-hop cost measurement on a shared GPU.  Contention on a shared GPU changes between seconds, so the
three components (speculator, cross-encoder, reader re-generation) are timed in round-robin over R rounds on the same
hop states, and the median of each component's per-round times is reported together with the per-round ratios.
Output: runs/<wl>/spec_bench2.json"""
import argparse, time
import numpy as np
from common import *
from mcert import MarginBook
from spec import Speculator
from deltar import load_dr, score_examples
from rag import HOP

ap = argparse.ArgumentParser()
ap.add_argument("--wl", default="wiki_t")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--rounds", type=int, default=7)
ap.add_argument("--specs", default="q06:spec_q06_d0,q17:spec_q17_d0")
ap.add_argument("--mtr", default="mtr:_s0")
ap.add_argument("--bs", type=int, default=16)
args = ap.parse_args()

R = pload(f"{RUNS}/{args.wl}/test/oracle.pkl")
MB = MarginBook(args.wl, "test")
texts = R["texts"]
states = [(q, rows, tuple(st["tok"]), st["ans"]) for (q, rows), st in MB.S.items() if st["tok"]]
rng = np.random.default_rng(0); rng.shuffle(states); states = states[: args.n]
print(args.wl, "states", len(states), flush=True)

comps = {}
for item in args.specs.split(","):
    name, ad = (item.split(":") + [""])[:2]
    sp = Speculator(name, ad)
    def f_spec(sp=sp):
        sp.memo.clear()
        sp.margins(texts, [(q, rows, y) for q, rows, y, _ in states], bs=args.bs)
    comps[f"spec_{name}{'+' + ad if ad else ''}"] = f_spec
if args.mtr:
    v, tag = (args.mtr.split(":") + [""])[:2]
    M = load_dr(v, tag)
    exs = [{"subq": q, "ans": a, "K": [(0, x) for x in rows], "K2": [(0, x) for x in rows], "mu": MB.mu(q, rows)} for q, rows, y, a in states]
    comps["mtr"] = lambda: score_examples(M, exs, {}, lambda d: texts[d[1]], bs=args.bs)
llm = LLMWrap(mem=float(os.environ.get("LLM_MEM", 0.5)))
tok = llm.tok
def prompt(q, rows):
    docs = "\n".join(f"[{i + 1}] {texts[r]}" for i, r in enumerate(rows))
    return tok.apply_chat_template([{"role": "user", "content": HOP.format(docs=docs, q=q)}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
SP = llm.SP(temperature=0, max_tokens=32, logprobs=20)
ps = [prompt(q, rows) for q, rows, _, _ in states]
comps["reader"] = lambda: llm.llm.generate(ps, SP, use_tqdm=False)

for f in comps.values():                      # warm-up
    f(); torch.cuda.synchronize()
times = {k: [] for k in comps}
for r in range(args.rounds):
    for k, f in comps.items():
        torch.cuda.synchronize(); t = time.time(); f(); torch.cuda.synchronize()
        times[k].append(1000 * (time.time() - t) / len(states))
    print("round", r, {k: round(v[-1], 1) for k, v in times.items()}, flush=True)
out = {"wl": args.wl, "n": len(states), "rounds": args.rounds, "gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?"), "session": time.strftime("%Y-%m-%d %H:%M")}
for k, v in times.items():
    out[k + "_ms_median"] = float(np.median(v)); out[k + "_ms_min"] = float(np.min(v)); out[k + "_ms_max"] = float(np.max(v))
for k in comps:
    if k != "reader":
        out[k + "_over_reader_median"] = float(np.median(np.array(times[k]) / np.array(times["reader"])))
jdump(out, f"{RUNS}/{args.wl}/spec_bench2.json")
print("BENCH2 DONE", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}, flush=True)
