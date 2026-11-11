#!/bin/bash
# Round-8 experiment queue: five lanes in parallel, every job placed by sched.sh; writes STAGE V20_ALLDONE at the end.
source /data/yaokang/evcache/code/sched.sh
st v20_start
laneA(){  # same-backbone controls: the delta reader (old + new versions only) and the before/after set reader
  job 9000 dr_setcrossD_m3 env DR_SETLEN=1024 $PY deltar.py --model setcrossD --epochs 2 --bs 8 --backbone $M3 --ckpt --tag _m3
  for w in wiki_t wiki_cf hotpot wiki_rev; do
    job 6000 ev11_$w $PY run_eval3.py --wl $w --dr setcrossD:_m3,setcross2:_m3 --v2 "" --skip_nli --skip_margin --out eval11; done
  job 6000 closedD_wiki_rev $PY run_closed.py --wl wiki_rev --scorers dr:setcrossD:_m3,dr:setcross2:_m3 --alphas 0.1 --pis 0.25 --out closed_delta
}
laneB(){  # closed loop (stale lifetime, filter time, numeric bound) + request logs, then cold / mis-calibrated starts
  for w in wiki_cf wiki_t; do job 7000 closed2_$w $PY run_closed.py --wl $w --dump; done
  for w in wiki_cf wiki_t; do job 5000 cold_$w $PY run_closed.py --wl $w --hist_from none --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cold; done
  job 5000 cross_wiki_cf $PY run_closed.py --wl wiki_cf --hist_from wiki_t --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross
  job 5000 cross_wiki_t $PY run_closed.py --wl wiki_t --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross
}
laneC(){
  job 7000 closed2_wiki_rev $PY run_closed.py --wl wiki_rev --dump --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3,set-cross1_wr
  job 7000 closed2_hotpot $PY run_closed.py --wl hotpot --dump
  for w in hotpot wiki_rev; do job 5000 cold_$w $PY run_closed.py --wl $w --hist_from none --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cold; done
  job 5000 cross_wiki_rev $PY run_closed.py --wl wiki_rev --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross
  job 5000 cross_hotpot $PY run_closed.py --wl hotpot --hist_from wiki_cf --scorers set-cross1_m3 --alphas 0.1 --pis 0.25 --out cross
}
laneD(){ for w in wiki_t wiki_cf hotpot wiki_rev; do job 2000 ann2_$w $PY ann_reach.py --wl $w; done; }
laneE(){  # end-to-end serving with the INT4 8B reader, once the request logs exist
  for w in wiki_cf wiki_rev; do
    while ! grep -q "^dumped" $L/closed2_$w.log 2>/dev/null; do grep -q "FAIL_closed2_$w" $L/stages_v20.log && break; sleep 120; done
    job 11000 serve_$w env LLM_SEQS=32 LLM_MAXLEN=4096 $PY serve_bench.py --wl $w
  done
}
laneA & laneB & laneC & laneD & laneE &
wait
st V20_ALLDONE
