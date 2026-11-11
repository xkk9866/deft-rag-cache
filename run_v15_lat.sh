#!/bin/bash
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs
until grep -q LL3DONE $L/stages_v9.log; do sleep 60; done
LLM_MEM=0.42 LLM_SEQS=32 /home/yaokang/research/qrag/.venv/bin/python latency.py --wl wiki_cf > $L/latency.log 2>&1
echo "STAGE LAT2_DONE $(date)" >> $L/stages_v9.log
