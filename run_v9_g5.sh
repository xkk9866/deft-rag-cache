#!/bin/bash
# GPU5: W3 re-evaluation, then strict answer equivalence for DEFT-X
export CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v9.log; }
DRS="setcross1:_m3,setcross2:_m3,setcrossE:_m3,full:_rr,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,full:_rr1,full:_rr2"
$PY run_eval3.py --wl hotpot --dr $DRS --v2 cross --out eval9 > $L/ev9_hotpot.log 2>&1; st g5_hotpot
for wl in wiki_t wiki_cf; do EQUIV=strict $PY run_eval3.py --wl $wl --skip_margin --dr setcross1:_m3 --v2 "" --skip_nli --out eval9strict > $L/ev9strict_$wl.log 2>&1; done
st G5DONE
