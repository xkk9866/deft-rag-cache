#!/bin/bash
# shorten the margin pass: training needs only the W1-W3 training margins; the 4B / Llama readers need test+cal only
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v30.log; }
while [ ! -f /data/yaokang/evcache/runs/hotpot/train/margins.pkl ]; do sleep 30; done
sleep 20
pkill -f '[r]un_v30_margins.sh'; sleep 2; pkill -f '[m]argins.py --jobs wiki_t:test'; sleep 20
st ok_margins_q8_trainW123
LLM_MEM=0.85 LLM_PATH=/home/yaokang/models/Qwen3-4B-AWQ $PY margins.py --jobs wiki_t_q4:test,wiki_t_q4:cal > $L/margins_q4.log 2>&1 && st ok_margins_q4 || st FAIL_margins_q4
LLM_MEM=0.85 LLM_PATH=/data/yaokang/evcache/models/llama31-8b-awq $PY margins.py --jobs wiki_t_ll:test,wiki_t_ll:cal > $L/margins_ll.log 2>&1 && st ok_margins_ll || st FAIL_margins_ll
st MARGINS_ALLDONE
