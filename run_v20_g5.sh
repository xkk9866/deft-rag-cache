#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v20.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
export CUDA_VISIBLE_DEVICES=5
( for w in wiki_t wiki_cf hotpot wiki_rev; do $PY ann_reach.py --wl $w > $L/ann2_$w.log 2>&1 || st FAIL_ann_$w; done; st ANN2_DONE ) &
$PY run_closed.py --wl wiki_rev --dump --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3,set-cross1_wr > $L/closed2_wiki_rev.log 2>&1 || st FAIL_closed_wiki_rev
st closed_wiki_rev
$PY run_closed.py --wl hotpot --dump > $L/closed2_hotpot.log 2>&1 || st FAIL_closed_hotpot
st closed_hotpot
for w in hotpot wiki_rev; do $PY run_closed.py --wl $w --hist_from none --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cold > $L/cold_$w.log 2>&1 || st FAIL_cold_$w; done
$PY run_closed.py --wl wiki_rev --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross > $L/cross_wiki_rev.log 2>&1 || st FAIL_cross_wiki_rev
$PY run_closed.py --wl hotpot --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross > $L/cross_hotpot.log 2>&1 || st FAIL_cross_hotpot
wait
st G5_DONE
