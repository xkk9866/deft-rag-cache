#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
for m in svde svde_noaux bienc cross; do $PY deft.py --model $m > ../logs/train_$m.log 2>&1 & done
wait
echo "STAGE train DONE $(date)" >> ../logs/stages.log
