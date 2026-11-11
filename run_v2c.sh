#!/bin/bash
# follow-up: after run_v2b finishes, W1 cal/test oracles sequentially (one vLLM engine at a time), then W1 eval
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
cd /data/yaokang/evcache/code
L=../logs
until grep -q ALLDONE $L/stages_v2.log; do sleep 60; done
for s in cal test; do LLM_MEM=0.6 $PY oracle.py --wl wiki_t --split $s --B 10 > $L/oracle_wiki_t_$s.log 2>&1; done
echo "STAGE oracle_t_calTest $(date)" >> $L/stages_v2.log
$PY run_eval.py --wl wiki_t > $L/eval_wiki_t.log 2>&1
echo "STAGE V2C_DONE $(date)" >> $L/stages_v2.log
