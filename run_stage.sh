#!/bin/bash
# usage: bash run_stage.sh <stage>   (all logs in ../logs)
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
L=../logs
orc(){ [ -f ../runs/$1/$2/oracle.pkl ] || $PY oracle.py --wl $1 --split $2 > $L/oracle_$1_$2.log 2>&1; }
case "$1" in
  oracle_t)  for s in cal test train; do orc wiki_t $s; done ;;
  hotpot)    [ -f ../data/hotpot/corpus.json ] || $PY build_hotpot.py > $L/build_hotpot.log 2>&1
             for s in cal test train; do orc hotpot $s; done ;;
  oracle_cf) for s in cal test train; do orc wiki_cf $s; done ;;
  label)     for w in wiki_t wiki_cf hotpot; do [ -f ../runs/$w/train/pairs.json ] || $PY label.py --wl $w > $L/label_$w.log 2>&1; done ;;
  train)     for m in svde svde_noaux bienc cross; do $PY deft.py --model $m > $L/train_$m.log 2>&1; done ;;
  eval)      for w in wiki_t wiki_cf hotpot; do $PY run_eval.py --wl $w > $L/eval_$w.log 2>&1; done ;;
  bench)     $PY bench.py --wl wiki_cf --latency > $L/bench.log 2>&1 ;;
esac
echo "STAGE $1 DONE $(date)" >> $L/stages.log
