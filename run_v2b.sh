#!/bin/bash
# v2 queue (resume) -- GPU 7 only
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v2.log; }
rm -rf ../runs/wiki_t
LLM_MEM=0.42 $PY oracle.py --wl wiki_t --split cal --B 10 > $L/oracle_wiki_t_cal.log 2>&1 &
LLM_MEM=0.42 $PY oracle.py --wl wiki_t --split test --B 10 > $L/oracle_wiki_t_test.log 2>&1 &
wait
LLM_MEM=0.6 $PY oracle.py --wl wiki_t --split train --B 10 > $L/oracle_wiki_t_train.log 2>&1; st oracle_t
$PY label.py --wl wiki_t > $L/label_wiki_t.log 2>&1; st label_t
$PY deft.py --model svdef --bs 32 > $L/train_svdef.log 2>&1
$PY deft.py --model svdef_noaux --bs 32 > $L/train_svdef_noaux.log 2>&1
$PY deft.py --model svde > $L/train_svde.log 2>&1 & $PY deft.py --model bienc > $L/train_bienc.log 2>&1 & wait
$PY deft.py --model cross > $L/train_cross.log 2>&1; st train
for w in wiki_t wiki_cf hotpot; do $PY run_eval.py --wl $w > $L/eval_$w.log 2>&1 & done; wait; st eval
$PY bench.py --wl wiki_cf --latency > $L/bench.log 2>&1; st bench
st ALLDONE
