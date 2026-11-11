#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v20.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
export CUDA_VISIBLE_DEVICES=7
for w in wiki_cf wiki_t; do $PY run_closed.py --wl $w --dump > $L/closed2_$w.log 2>&1 || st FAIL_closed_$w; st closed_$w; done
for w in wiki_cf wiki_t; do $PY run_closed.py --wl $w --hist_from none --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cold > $L/cold_$w.log 2>&1 || st FAIL_cold_$w; done
$PY run_closed.py --wl wiki_cf --hist_from wiki_t --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross > $L/cross_wiki_cf.log 2>&1 || st FAIL_cross_wiki_cf
$PY run_closed.py --wl wiki_t --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross > $L/cross_wiki_t.log 2>&1 || st FAIL_cross_wiki_t
st G7_DONE
