#!/bin/bash
# v7 (GPU 7 only): DEFT-X ablations on the same backbone, distillation of DEFT-X into DEFT-L, evaluation, bench
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v7.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
st start
$PY deltar.py --model full --epochs 3 --reads --rank --teacher setcross2:_m3 --tag _kd > $L/dr_full_kd.log 2>&1; st kd
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --backbone $M3 --ckpt --tag _m3 > $L/dr_setcross1_m3.log 2>&1; st x1
DR_SETLEN=1024 $PY deltar.py --model setcrossE --epochs 2 --backbone $M3 --ckpt --tag _m3 > $L/dr_setcrossE_m3.log 2>&1; st xE
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full:_kd,setcross1:_m3,setcrossE:_m3 --v2 "" --skip_nli --skip_margin --out eval7 > $L/ev7_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t_q4 --dr full:_kd,setcross2:_m3 --v2 "" --skip_nli --skip_margin --out eval7 > $L/ev7_q4.log 2>&1 &
wait; st eval
$PY bench3.py --wl wiki_cf > $L/bench3_v7.log 2>&1
st V7DONE
