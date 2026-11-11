#!/bin/bash
source /data/yaokang/evcache/code/sched.sh
st v25_start
for w in wiki_rev wiki_cf; do job 13000 serve_$w env LLM_MEM=0.45 LLM_SEQS=16 LLM_MAXLEN=3072 $PY serve_bench.py --wl $w --rates 200,400; done
st V25_ALLDONE
