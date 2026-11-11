#!/bin/bash
# v2 experiment queue -- GPU 7 only
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v2.log; }
# 1. W1 re-split (more calibration batches) + oracles with batch size 10
rm -rf ../runs/wiki_t ../data/wiki_t/split_*.json
$PY build_mquake.py --wl wiki_t > $L/build_wiki_t_v2.log 2>&1; st build_t
for s in cal test train; do LLM_MEM=0.28 $PY oracle.py --wl wiki_t --split $s --B 10 > $L/oracle_wiki_t_$s.log 2>&1 & done; wait; st oracle_t
$PY label.py --wl wiki_t > $L/label_wiki_t.log 2>&1; st label_t
# 2. models (at most 2 concurrent trainings)
$PY deft.py --model svdef > $L/train_svdef.log 2>&1 & $PY deft.py --model svdef_noaux > $L/train_svdef_noaux.log 2>&1 & wait
$PY deft.py --model svde > $L/train_svde.log 2>&1 & $PY deft.py --model bienc > $L/train_bienc.log 2>&1 & wait
$PY deft.py --model cross > $L/train_cross.log 2>&1; st train
# 3. evaluation (3 workloads in parallel) and benchmarks
for w in wiki_t wiki_cf hotpot; do $PY run_eval.py --wl $w > $L/eval_$w.log 2>&1 & done; wait; st eval
$PY bench.py --wl wiki_cf --latency > $L/bench.log 2>&1; st bench
st ALLDONE
