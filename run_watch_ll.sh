#!/bin/bash
# waits until one of GPUs 1/5/7 has >= 13 GB free, then runs the Llama reader pipeline and the latency measurement there
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v9.log; }
G=""
while [ -z "$G" ]; do
  for g in 1 5 7; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    if [ "$free" -ge 13000 ]; then G=$g; break; fi
  done
  [ -z "$G" ] && sleep 120
done
st watch_gpu_$G
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LLM_MEM=0.45 LLM_SEQS=32 $PY latency.py --wl wiki_cf > $L/latency.log 2>&1; st watch_lat
RD=$R/models/llama31-8b-awq
for sp in train cal test; do LLM_PATH=$RD LLM_MEM=0.5 LLM_SEQS=64 $PY oracle.py --wl wiki_t_ll --split $sp --B 10 > $L/oracle_ll_$sp.log 2>&1; done
LLM_PATH=$RD LLM_MEM=0.5 LLM_SEQS=64 $PY label.py --wl wiki_t_ll > $L/label_ll.log 2>&1
$PY run_eval.py --wl wiki_t_ll --models "" > $L/ev_ll_base.log 2>&1
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --wls wiki_t_ll --init dr_setcross1_m3 --backbone $(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*) --ckpt --tag _ll > $L/dr_setcross1_ll.log 2>&1
$PY run_eval3.py --wl wiki_t_ll --extra --dr setcross1:_m3,setcross1:_ll,full:_rr --v2 "" --out eval9 > $L/ev9_ll.log 2>&1
st WATCH_DONE
