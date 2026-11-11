#!/bin/bash
# v8 (GPU 7 only, shared with another user's ~9.5GB process): SEQUENTIAL evaluations, provenance-based before-sets
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v8.log; }
st start_seq
DRS="setcross1:_m3,setcross2:_m3,setcrossE:_m3,full:_kd,full:_rr,full:_rr320,full:_nli,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,full:,setcross:,full:_rr1,full:_rr2"
for wl in wiki_cf wiki_t hotpot; do $PY run_eval3.py --wl $wl --extra --skip_margin --dr $DRS --v2 "" --out eval8 > $L/ev8_$wl.log 2>&1; st ev8_$wl; done
$PY run_eval3.py --wl wiki_t_q4 --extra --skip_margin --dr setcross1:_m3,setcross2:_m3,full:_kd,full:_rr,full:_q4 --v2 "" --out eval8 > $L/ev8_q4.log 2>&1; st ev8_q4
$PY bench3.py --wl wiki_cf > $L/bench3_v8.log 2>&1
st V8DONE
