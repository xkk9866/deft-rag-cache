#!/bin/bash
# Delta-Reader with the same per-document token budget as the NLI baseline (320)
export CUDA_VISIBLE_DEVICES=7 DR_DOCLEN=320
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v5.log; }
st start
$PY deltar.py --model full --epochs 3 --reads --rank --tag _rr320 > $L/dr_full_rr320.log 2>&1 || $PY deltar.py --model full --epochs 3 --reads --rank --bs 12 --tag _rr320 > $L/dr_full_rr320.log 2>&1
st train
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full:_rr320 --v2 "" --skip_nli --out eval5 > $L/ev5_$wl.log 2>&1 & done; wait
st V5DONE
