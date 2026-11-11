#!/bin/bash
# final pipeline (GPU 7 only): ablations of DR_rr under the same objective, seeds, transfer, all evaluations, bench
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v4.log; }
tr(){ $PY deltar.py --model $1 --epochs 3 $2 > $L/dr_$1$3.log 2>&1; }
st start
tr noset "--reads --rank --tag _rr" _rr &
while pgrep -f "[d]eltar.py --model doclevel" > /dev/null; do sleep 20; done
tr noaux "--rank --tag _rr" _rr & wait
tr full "--reads --tag _r" _r & tr full "--reads --rank --seed 1 --tag _rr1" _rr1 & wait; st trainR2
DRS="full:_rr,full:_r,noset:_rr,doclevel:_rr,noaux:_rr,setcross:,full:"
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr $DRS --v2 svdef,cross --out eval4 > $L/ev4_$wl.log 2>&1 & done
tr full "--reads --rank --seed 2 --tag _rr2" _rr2; tr full "--reads --rank --wls wiki_cf,hotpot --tag _rrx" _rrx
wait; st eval
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full:_rr1,full:_rr2 --v2 "" --skip_nli --out eval4s > $L/ev4s_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t --dr full:_rrx --v2 "" --skip_nli --out eval4x > $L/ev4x.log 2>&1 &
wait; st evalS
$PY bench3.py --wl wiki_cf > $L/bench3.log 2>&1
st V4DONE
