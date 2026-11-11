#!/bin/bash
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
for w in wiki_t wiki_cf hotpot; do /home/yaokang/research/qrag/.venv/bin/python run_closed.py --wl $w > ../logs/closed_$w.log 2>&1 || echo "STAGE CLOSED_FAIL_$w $(date)" >> ../logs/stages_v19.log; echo "STAGE closed_$w $(date)" >> ../logs/stages_v19.log; done
