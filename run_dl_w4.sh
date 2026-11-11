#!/bin/bash
# W4 data: locate and read the W1 entities in two REAL Wikipedia dumps by byte-range access, then run the W4 pipeline
cd /data/yaokang/evcache/code
L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v18.log; }
mkdir -p /data/yaokang/evcache/data/wiki_rev
st w4_scan_start
/home/yaokang/research/qrag/.venv/bin/python scan_wikidump.py 24 64 > $L/dl_w4.log 2>&1 || { st W4_FAIL_scan; exit 1; }
st w4_scan_done
bash run_w4.sh ${1:-0}
