#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
/home/yaokang/research/qrag/.venv/bin/python bench_indep.py --wl wiki_cf --max_host_gb 24 > ../logs/bench_indep.log 2>&1
echo "STAGE BENCH_INDEP $(date)" >> ../logs/stages_v18.log
