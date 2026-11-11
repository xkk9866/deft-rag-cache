#!/bin/bash
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=../logs
LLM_MEM=0.30 LLM_SEQS=4 LLM_EAGER=1 LLM_MAXLEN=3072 LLM_BTOK=3072 /home/yaokang/research/qrag/.venv/bin/python latency.py --wl wiki_cf > $L/latency.log 2>&1
echo "STAGE LAT3_DONE $(date)" >> $L/stages_v9.log
