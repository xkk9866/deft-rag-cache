#!/bin/bash
# v6 queue (GPU 7 only): reviewer round 4 -- second reader, NLI-backbone Delta-Reader, long-context set cross-encoder,
# recent baselines (reader-judge, FreshCache-style, set-NLI), refreshed-state replay, temporal validity, sensitivity.
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v6.log; }
LLAMA=/home/yaokang/models/Qwen2.5-7B-Instruct-AWQ
NLIB=$R/models/deberta-v3-large-nli
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
st start
mkdir -p $R/data/wiki_t_q25; for f in $R/data/wiki_t/*; do ln -sf $f $R/data/wiki_t_q25/; done
llm_chain(){
  for sp in train cal test; do LLM_PATH=$LLAMA LLM_MEM=0.45 $PY oracle.py --wl wiki_t_q25 --split $sp --B 10 > $L/oracle_q25_$sp.log 2>&1; done
  LLM_PATH=$LLAMA LLM_MEM=0.45 $PY label.py --wl wiki_t_q25 > $L/label_q25.log 2>&1
  LLM_PATH=$LLAMA LLM_MEM=0.45 $PY label_reads.py --wls wiki_t_q25 > $L/reads_q25.log 2>&1
  st llm_chain
}
train_chain(){
  $PY deltar.py --model full --epochs 3 --reads --rank --backbone $NLIB --ckpt --tag _nli > $L/dr_full_nli.log 2>&1
  DR_SETLEN=1024 $PY deltar.py --model setcross2 --epochs 2 --backbone $M3 --ckpt --tag _m3 > $L/dr_setcross2_m3.log 2>&1
  st train_chain
}
llm_chain & train_chain & wait; st A
$PY run_eval.py --wl wiki_t_q25 --models "" > $L/ev_q25_base.log 2>&1 &
$PY deltar.py --model full --epochs 3 --reads --rank --wls wiki_t_q25 --init dr_full_rr --tag _q25 > $L/dr_full_q25.log 2>&1
wait; st B
DRS="full:_rr,full:_nli,setcross2:_m3,full:_rr320,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,setcross:,full:"
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --extra --dr $DRS --v2 svdef,cross --out eval6 > $L/ev6_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t_q25 --extra --dr full:_rr,full:_q25 --v2 "" --out eval6 > $L/ev6_q25.log 2>&1 &
wait; st C
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full:_rr1,full:_rr2 --v2 "" --skip_nli --skip_margin --out eval6s > $L/ev6s_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t --dr full:_rrx --v2 "" --skip_nli --skip_margin --out eval6x > $L/ev6x.log 2>&1 &
wait
for wl in wiki_t wiki_cf hotpot; do DR_INT8=1 $PY run_eval3.py --wl $wl --dr full:_rr --v2 "" --skip_nli --skip_margin --out eval6q8 > $L/ev6q8_$wl.log 2>&1 & done
for wl in wiki_t wiki_cf; do EQUIV=strict $PY run_eval3.py --wl $wl --dr full:_rr --v2 cross --out eval6strict > $L/ev6strict_$wl.log 2>&1 & done
wait; st D
$PY bench3.py --wl wiki_cf > $L/bench3_v6.log 2>&1
st V6DONE
