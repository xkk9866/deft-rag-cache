#!/bin/bash
export CUDA_VISIBLE_DEVICES=5 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/home/yaokang/research/qrag/.venv/bin/python run_eval3.py --wl hotpot --dr setcross1:_m3,full:_rr --v2 cross --out eval10 --splits 50 > ../logs/ev10g5_hotpot.log 2>&1
echo "STAGE G5B_DONE $(date)" >> ../logs/stages_v9.log
