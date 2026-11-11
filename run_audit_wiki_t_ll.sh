#!/bin/bash
export CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
/home/yaokang/research/qrag/.venv/bin/python run_audit.py --wl wiki_t_ll --scorers margin,nli,DR_rr,set-cross1_m3,set-cross1_ll > ../logs/audit_wiki_t_ll.log 2>&1
echo "STAGE AUDIT_wiki_t_ll $(date)" >> ../logs/stages_v18.log
