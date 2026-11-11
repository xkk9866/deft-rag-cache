#!/bin/bash
# GPU7: single-request reader latency (interactive cost of one reader call)
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
until grep -q G7B_DONE $L/stages_v9.log 2>/dev/null; do sleep 60; done
LLM_MEM=0.5 $PY latency.py --wl wiki_cf > $L/latency.log 2>&1
echo "STAGE LAT_DONE $(date)" >> $L/stages_v9.log
