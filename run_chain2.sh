#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
for m in svde svde_noaux; do $PY deft.py --model $m > ../logs/train_$m.log 2>&1 & done
wait
echo "STAGE train2 DONE $(date)" >> ../logs/stages.log
bash run_eval_all.sh
