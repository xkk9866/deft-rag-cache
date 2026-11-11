#!/bin/bash
# round 9 timing lane: dedicated GPU 6 after all training/evaluation (no other job on the card)
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True VLLM_USE_V1=0
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v31.log; }
while ! { grep -q TRAIN_ALLDONE $L/stages_v31.log && grep -q EVAL_ALLDONE $L/stages_v31.log; }; do sleep 60; done
for w in wiki_t wiki_cf hotpot wiki_rev; do LLM_MEM=0.85 $PY verify_bench.py --wl $w > $L/vb_$w.log 2>&1 && st ok_vb_$w || st FAIL_vb_$w; done
for w in wiki_cf wiki_rev; do LLM_MEM=0.75 $PY serve_bench2.py --wl $w > $L/sb2_$w.log 2>&1 && st ok_sb2_$w || st FAIL_sb2_$w; done
st BENCH_ALLDONE
