#!/bin/bash
export CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
/home/yaokang/research/qrag/.venv/bin/python run_audit.py --wl wiki_t --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3 > ../logs/audit_wiki_t.log 2>&1
echo "STAGE AUDIT_wiki_t $(date)" >> ../logs/stages_v18.log
