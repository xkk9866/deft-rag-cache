#!/bin/bash
# GPU7: re-evaluation with certification against the last trusted computation (W2, W1)
export CUDA_VISIBLE_DEVICES=7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v9.log; }
DRS="setcross1:_m3,setcross2:_m3,setcrossE:_m3,full:_rr,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,full:_rr1,full:_rr2"
for wl in wiki_cf wiki_t; do $PY run_eval3.py --wl $wl --dr $DRS --v2 cross --out eval9 > $L/ev9_$wl.log 2>&1; st g7_$wl; done
st G7DONE
