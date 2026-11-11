#!/bin/bash
# GPU1: non-Qwen reader (Llama-3.1-8B-Instruct AWQ-INT4) on W1, end to end
export CUDA_VISIBLE_DEVICES=5
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v9.log; }
st g5ll_download
RD=$R/models/llama31-8b-awq
mkdir -p $R/data/wiki_t_ll; for f in $R/data/wiki_t/*; do ln -sf $f $R/data/wiki_t_ll/; done
for sp in train cal test; do LLM_PATH=$RD LLM_MEM=0.33 LLM_SEQS=32 $PY oracle.py --wl wiki_t_ll --split $sp --B 10 > $L/oracle_ll_$sp.log 2>&1; done
LLM_PATH=$RD LLM_MEM=0.33 LLM_SEQS=32 $PY label.py --wl wiki_t_ll > $L/label_ll.log 2>&1
st g5ll_llm
$PY run_eval.py --wl wiki_t_ll --models "" > $L/ev_ll_base.log 2>&1
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --wls wiki_t_ll --init dr_setcross1_m3 --backbone $(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*) --ckpt --tag _ll > $L/dr_setcross1_ll.log 2>&1
$PY run_eval3.py --wl wiki_t_ll --extra --dr setcross1:_m3,setcross1:_ll,full:_rr --v2 "" --out eval9 > $L/ev9_ll.log 2>&1
st LLDONE
