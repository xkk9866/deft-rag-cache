#!/bin/bash
# v3 pipeline (GPU 7 only): Delta-Reader training + ablations + evaluation
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v3.log; }
tr(){ $PY deltar.py --model $1 --epochs 3 $2 > $L/dr_$1$3.log 2>&1; }
st start
tr full "" "" & tr noset "" "" & wait; st trainA
tr doclevel "" "" & tr noaux "" "" & wait; st trainB
tr setcross "" "" & tr full "--wls wiki_cf,hotpot --tag _xfer" _xfer & wait; st trainC
tr full "--seed 1 --tag _s1" _s1 & tr full "--seed 2 --tag _s2" _s2 & wait; st trainD
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl > $L/ev3_$wl.log 2>&1 & done; wait; st eval
$PY run_eval3.py --wl wiki_t --dr full --v2 "" --skip_nli --tag _xfer --out eval3x > $L/ev3x.log 2>&1 &
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full --v2 "" --skip_nli --tag _s1 --out eval3s1 --splits 50 > $L/ev3s1_$wl.log 2>&1 & done; wait
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full --v2 "" --skip_nli --tag _s2 --out eval3s2 --splits 50 > $L/ev3s2_$wl.log 2>&1 & done; wait
st V3DONE
