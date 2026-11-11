#!/bin/bash
# second lane of v8: W3 and W1/4B (main lane: W2, W1)
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v8.log; }
DRS="setcross1:_m3,setcross2:_m3,setcrossE:_m3,full:_kd,full:_rr,full:_rr320,full:_nli,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,full:,setcross:,full:_rr1,full:_rr2"
$PY run_eval3.py --wl hotpot --extra --skip_margin --dr $DRS --v2 "" --out eval8 > $L/ev8b_hotpot.log 2>&1; st ev8b_hotpot
$PY run_eval3.py --wl wiki_t_q4 --extra --skip_margin --dr setcross1:_m3,setcross2:_m3,full:_kd,full:_rr,full:_q4 --v2 "" --out eval8 > $L/ev8b_q4.log 2>&1; st ev8b_q4
