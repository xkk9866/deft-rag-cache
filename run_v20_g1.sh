#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v20.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
export CUDA_VISIBLE_DEVICES=1 LLM_SEQS=32 LLM_MAXLEN=4096
R=/data/yaokang/evcache/runs
waitmem(){ while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1) -gt 14500 ]; do sleep 60; done; sleep 10; }
for w in wiki_cf wiki_rev; do
  while [ ! -s $R/$w/closed/reqlog.pkl ] || ! grep -q "^dumped" $L/closed2_$w.log 2>/dev/null; do sleep 60; done
  waitmem
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  LLM_MEM=$(python3 -c "print(min(0.95, ($used + 9500) / 24564))") $PY serve_bench.py --wl $w > $L/serve_$w.log 2>&1 || st FAIL_serve_$w
  st serve_$w
done
st G1_DONE
