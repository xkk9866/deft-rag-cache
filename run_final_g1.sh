#!/bin/bash
# GPU1: latency + Llama-3.1-8B reader pipeline. vLLM's memory fraction counts other processes, so compute it from current usage.
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v9.log; }
memfrac(){ used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1); python3 -c "print(min(0.95, ($used + 11500) / 24564))"; }
st final_start
LLM_MEM=$(memfrac) LLM_SEQS=32 $PY latency.py --wl wiki_cf > $L/latency.log 2>&1; st final_lat
RD=$R/models/llama31-8b-awq
for sp in train cal test; do LLM_PATH=$RD LLM_MEM=$(memfrac) LLM_SEQS=64 $PY oracle.py --wl wiki_t_ll --split $sp --B 10 > $L/oracle_ll_$sp.log 2>&1; done
LLM_PATH=$RD LLM_MEM=$(memfrac) LLM_SEQS=64 $PY label.py --wl wiki_t_ll > $L/label_ll.log 2>&1; st final_llm
$PY run_eval.py --wl wiki_t_ll --models "" > $L/ev_ll_base.log 2>&1
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --wls wiki_t_ll --init dr_setcross1_m3 --backbone $(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*) --ckpt --tag _ll > $L/dr_setcross1_ll.log 2>&1
$PY run_eval3.py --wl wiki_t_ll --extra --dr setcross1:_m3,setcross1:_ll,full:_rr --v2 "" --out eval9 > $L/ev9_ll.log 2>&1
st LLFINAL_DONE
