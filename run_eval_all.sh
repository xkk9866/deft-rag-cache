#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
TAG=${1:-}
for w in wiki_t wiki_cf hotpot; do $PY run_eval.py --wl $w ${TAG:+--tag $TAG} > ../logs/eval_$w$TAG.log 2>&1 & done
wait
echo "STAGE eval$TAG DONE $(date)" >> ../logs/stages.log
