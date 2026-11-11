#!/bin/bash
# parallel lane: W2 oracles with a smaller vLLM memory share on the same GPU
export CUDA_VISIBLE_DEVICES=7 LLM_MEM=0.34
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
for s in cal test train; do [ -f ../runs/wiki_cf/$s/oracle.pkl ] || $PY oracle.py --wl wiki_cf --split $s > ../logs/oracle_wiki_cf_$s.log 2>&1; done
echo "STAGE oracle_cf_parallel DONE $(date)" >> ../logs/stages.log
