#!/bin/bash
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
/home/yaokang/research/qrag/.venv/bin/python run_audit.py --wl wiki_cf --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3 > ../logs/audit_wiki_cf.log 2>&1
echo "STAGE AUDIT2_wiki_cf $(date)" >> ../logs/stages_v18.log
