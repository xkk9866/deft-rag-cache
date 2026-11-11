#!/bin/bash
# round 9 training lane (GPU 6 only): margin-transport readers, same-data controls, seeds
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v31.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
tr(){ n=$1; shift; DR_SETLEN=1024 $PY deltar.py --epochs 2 --backbone $M3 --ckpt --lowmem --micro 4 "$@" > $L/tr_$n.log 2>&1 && st ok_tr_$n || st FAIL_tr_$n; }
tr mtr_s0  --model mtr  --tag _s0 --seed 0
tr x_aug   --model setcross1 --aug --tag _aug --seed 0
tr mtr0_s0 --model mtr0 --tag _s0 --seed 0
tr mtr_s1  --model mtr  --tag _s1 --seed 1
tr mtr_s2  --model mtr  --tag _s2 --seed 2
tr mtrE_s0 --model mtrE --tag _s0 --seed 0
st TRAIN_ALLDONE
