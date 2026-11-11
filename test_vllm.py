import time, sys
from vllm import LLM, SamplingParams
m = sys.argv[1]
llm = LLM(model=m, quantization="awq_marlin", gpu_memory_utilization=0.55, max_model_len=4096, enforce_eager=False, seed=0)
tok = llm.get_tokenizer()
qs = ["Who is the current head of government of New York City?", "What is the capital of Australia?"] * 64
prompts = [tok.apply_chat_template([{"role": "user", "content": q + " Answer with a short phrase."}], tokenize=False, add_generation_prompt=True, enable_thinking=False) for q in qs]
sp = SamplingParams(temperature=0, max_tokens=32)
t = time.time(); outs = llm.generate(prompts, sp); dt = time.time() - t
print("time", dt, "per-sec", len(prompts) / dt)
for o in outs[:2]: print(repr(o.outputs[0].text))
