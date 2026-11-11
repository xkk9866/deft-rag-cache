#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v20.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
export CUDA_VISIBLE_DEVICES=4
DR_SETLEN=1024 $PY deltar.py --model setcrossD --epochs 2 --backbone $M3 --ckpt --tag _m3 > $L/dr_setcrossD_m3.log 2>&1 || st FAIL_trainD
st trainD
for w in wiki_t wiki_cf hotpot wiki_rev; do $PY run_eval3.py --wl $w --dr setcrossD:_m3,setcross2:_m3 --v2 "" --skip_nli --skip_margin --out eval11 > $L/ev11_$w.log 2>&1 || st FAIL_ev11_$w; st ev11_$w; done
$PY run_closed.py --wl wiki_rev --scorers dr:setcrossD:_m3,dr:setcross2:_m3 --alphas 0.1 --pis 0.25 --out closed_delta > $L/closedD_wiki_rev.log 2>&1 || st FAIL_closedD
st G4_DONE
