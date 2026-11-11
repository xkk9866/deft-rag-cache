#!/bin/bash
# reruns of the round-8 jobs that failed for memory; V0 engines size gpu_memory_utilization per instance
source /data/yaokang/evcache/code/sched.sh
st v21_start
( job 8000 ev11b_wiki_rev $PY run_eval3.py --wl wiki_rev --dr setcrossD:_m3,setcross2:_m3 --v2 "" --skip_nli --skip_margin --out eval11 ) &
( for w in wiki_cf wiki_rev; do job 12000 serve_$w env LLM_MEM=0.45 LLM_SEQS=32 LLM_MAXLEN=4096 $PY serve_bench.py --wl $w; done ) &
wait
st V21_ALLDONE
