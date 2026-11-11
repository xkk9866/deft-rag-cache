#!/bin/bash
# run a command that starts a vLLM engine on a shared GPU: LLM_MEM is the fraction of the whole card (vLLM semantics),
# computed from the current usage of the visible GPU plus the engine's need plus head-room for neighbours that grow.
# usage: with_llm.sh <need_MB> <command...>
need=$1; shift
g=${CUDA_VISIBLE_DEVICES:-6}
read used total <<< $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits -i $g | tr -d ',')
# vLLM fills its whole budget with KV cache, so the budget is exactly what the engine may take; head-room for the
# speculator and for neighbours that grow is the scheduler's job (declared task memory > need)
frac=$(python3 -c "print(min(0.95, round(($used + $need + 400) / $total, 3)))")
echo "gpu $g used $used total $total need $need -> LLM_MEM=$frac"
export LLM_MEM=$frac
export LLM_EAGER=${LLM_EAGER:-1} LLM_BTOK=${LLM_BTOK:-2048} LLM_MAXLEN=${LLM_MAXLEN:-3072} LLM_SEQS=${LLM_SEQS:-16}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec "$@"
