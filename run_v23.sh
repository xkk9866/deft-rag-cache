#!/bin/bash
source /data/yaokang/evcache/code/sched.sh
st v23_start
( for w in wiki_t wiki_rev; do job 6000 sweep10b_$w $PY run_closed.py --wl $w --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --sweep 10 --out sweep10; done ) &
( for w in wiki_cf wiki_rev; do job 12000 serve_$w env LLM_MEM=0.45 LLM_SEQS=32 LLM_MAXLEN=4096 $PY serve_bench.py --wl $w; done ) &
wait
st V23_ALLDONE
