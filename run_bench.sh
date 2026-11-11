#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
cd /data/yaokang/evcache/code
until grep -q V2C_DONE ../logs/stages_v2.log; do sleep 60; done
/home/yaokang/research/qrag/.venv/bin/python bench.py --wl wiki_cf --latency > ../logs/bench.log 2>&1
echo "STAGE BENCH_DONE $(date)" >> ../logs/stages_v2.log
