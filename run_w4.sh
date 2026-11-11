#!/bin/bash
# W4 (wiki_rev): real Wikipedia revision stream. Usage: bash run_w4.sh <gpu>
G=${1:-1}
export CUDA_VISIBLE_DEVICES=$G PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v18.log; }
memfrac(){ used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G); python3 -c "print(min(0.95, ($used + 11500) / 24564))"; }
st w4_start
$PY build_wikirev.py > $L/build_w4.log 2>&1 || { st W4_FAIL_build; exit 1; }
mf(){ used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1); python3 -c "print(min(0.95, ($used + 11500) / 24564))"; }
CUDA_VISIBLE_DEVICES=0 LLM_MEM=$(mf 0) LLM_SEQS=64 $PY oracle.py --wl wiki_rev --split test --B 20 > $L/oracle_w4_test.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=4 LLM_MEM=$(mf 4) LLM_SEQS=64 $PY oracle.py --wl wiki_rev --split train --B 20 > $L/oracle_w4_train.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=1 LLM_MEM=$(mf 1) LLM_SEQS=64 $PY oracle.py --wl wiki_rev --split cal --B 20 > $L/oracle_w4_cal.log 2>&1
wait $P1 $P2
for sp in test cal train; do [ -s $R/runs/wiki_rev/$sp/oracle.pkl ] || { st W4_FAIL_oracle_$sp; exit 1; }; done
st w4_oracle
LLM_MEM=$(memfrac) LLM_SEQS=64 $PY label.py --wl wiki_rev > $L/label_w4.log 2>&1
$PY run_eval.py --wl wiki_rev --models "" > $L/ev_w4_base.log 2>&1 || { st W4_FAIL_base; exit 1; }
st w4_base
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --wls wiki_rev --init dr_setcross1_m3 --backbone $(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*) --ckpt --tag _wr > $L/dr_setcross1_wr.log 2>&1
$PY run_eval3.py --wl wiki_rev --extra --dr setcross1:_m3,setcross1:_wr,full:_rr --v2 cross --out eval9 > $L/ev9_w4.log 2>&1 || { st W4_FAIL_eval; exit 1; }
st w4_eval
$PY run_audit.py --wl wiki_rev --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3,set-cross1_wr > $L/audit_wiki_rev.log 2>&1
st W4_DONE
