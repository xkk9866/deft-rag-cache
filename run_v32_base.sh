#!/bin/bash
# round 10: published baselines in the request-level closed loop (GPU 6 only), two workloads at a time
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v32.log; }
while ! grep -q DL_DONE $L/dl_base.log 2>/dev/null; do sleep 60; done
bl(){ $PY run_closed2.py --wl $1 --scorers "" --baselines gptcache,ttl,provenance,minicheck,hhem,nli_hit,vcache,vcache_g --seeds 5 --out base > $L/base_$1.log 2>&1 && st ok_base_$1 || st FAIL_base_$1; }
bl wiki_t & bl wiki_cf; wait
bl hotpot & bl wiki_rev; wait
st BASE_ALLDONE
