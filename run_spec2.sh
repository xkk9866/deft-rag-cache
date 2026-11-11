#!/bin/bash
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
for w in hotpot wiki_rev; do LLM_MEM=0.6 LLM_SEQS=8 LLM_MAXLEN=3072 LLM_BTOK=1024 LLM_EAGER=1 /home/yaokang/research/qrag/.venv/bin/python spec_cert.py --wl $w > ../logs/spec_$w.log 2>&1 || echo "STAGE SPEC2_FAIL_$w $(date)" >> ../logs/stages_v19.log; echo "STAGE spec2_$w $(date)" >> ../logs/stages_v19.log; done
echo "STAGE SPEC2_DONE $(date)" >> ../logs/stages_v19.log
