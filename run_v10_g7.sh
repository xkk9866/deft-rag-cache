#!/bin/bash
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
until grep -q G7DONE $L/stages_v9.log 2>/dev/null; do sleep 60; done
for wl in wiki_cf; do $PY run_eval3.py --wl $wl --dr setcross1:_m3,full:_rr --v2 cross --out eval10 --splits 50 > $L/ev10_$wl.log 2>&1; done
echo "STAGE G7B_DONE $(date)" >> $L/stages_v9.log
