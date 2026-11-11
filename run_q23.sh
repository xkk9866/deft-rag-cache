#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
for wl in wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full,noset,doclevel,noaux,setcross --v2 svdef --out eval3q --splits 50 > ../logs/ev3q_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t --dr setcross --v2 "" --skip_nli --out eval3q_sx --splits 50 > ../logs/ev3q_sx.log 2>&1 &
wait
echo "STAGE Q23 $(date)" >> ../logs/stages_v3.log
