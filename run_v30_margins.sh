#!/bin/bash
# round 9: reader margins (GPU 6 only)
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v30.log; }
LLM_MEM=0.85 $PY margins.py --jobs wiki_t:test,wiki_t:cal,wiki_cf:test,wiki_cf:cal,hotpot:test,hotpot:cal,wiki_rev:test,wiki_rev:cal,wiki_t:train,wiki_cf:train,hotpot:train,wiki_rev:train > $L/margins_q8.log 2>&1 && st ok_margins_q8 || st FAIL_margins_q8
LLM_MEM=0.85 LLM_PATH=/home/yaokang/models/Qwen3-4B-AWQ $PY margins.py --jobs wiki_t_q4:test,wiki_t_q4:cal,wiki_t_q4:train > $L/margins_q4.log 2>&1 && st ok_margins_q4 || st FAIL_margins_q4
LLM_MEM=0.85 LLM_PATH=/data/yaokang/evcache/models/llama31-8b-awq $PY margins.py --jobs wiki_t_ll:test,wiki_t_ll:cal,wiki_t_ll:train > $L/margins_ll.log 2>&1 && st ok_margins_ll || st FAIL_margins_ll
st MARGINS_ALLDONE
