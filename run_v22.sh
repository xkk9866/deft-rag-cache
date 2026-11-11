#!/bin/bash
# periodic speculative sweep: bounds the lifetime of a missed defeat by SWEEP batches; measured cost
source /data/yaokang/evcache/code/sched.sh
st v22_start
for P in 10 30; do for w in wiki_t wiki_cf hotpot wiki_rev; do
  ( job 5000 sweep${P}_$w $PY run_closed.py --wl $w --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --sweep $P --out sweep$P ) &
  sleep 5; done; done
wait
st V22_ALLDONE
