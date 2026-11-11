#!/bin/bash
# v6 queue, memory-safe order (GPU 7 only). vLLM always runs alone.
export CUDA_VISIBLE_DEVICES=7
PY=/home/yaokang/research/qrag/.venv/bin/python
L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v6.log; }
RD=/home/yaokang/models/Qwen3-4B-AWQ
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)
st start_v6c
while pgrep -f "[d]eltar.py --model full --epochs 3 --reads --rank --backbone" > /dev/null; do sleep 30; done; st train_nli
for sp in train cal test; do LLM_PATH=$RD LLM_MEM=0.6 $PY oracle.py --wl wiki_t_q4 --split $sp --B 10 > $L/oracle_q4_$sp.log 2>&1; done
LLM_PATH=$RD LLM_MEM=0.6 $PY label.py --wl wiki_t_q4 > $L/label_q4.log 2>&1
LLM_PATH=$RD LLM_MEM=0.6 $PY label_reads.py --wls wiki_t_q4 > $L/reads_q4.log 2>&1
st llm_chain
DR_SETLEN=1024 $PY deltar.py --model setcross2 --epochs 2 --backbone $M3 --ckpt --tag _m3 > $L/dr_setcross2_m3.log 2>&1 &
$PY run_eval.py --wl wiki_t_q4 --models "" > $L/ev_q4_base.log 2>&1 &
wait; st S3
DRS="full:_rr,full:_nli,setcross2:_m3,full:_rr320,noset:_rr,doclevel:_rr,noaux:_rr,full:_r,setcross:,full:"
$PY deltar.py --model full --epochs 3 --reads --rank --wls wiki_t_q4 --init dr_full_rr --tag _q4 > $L/dr_full_q4.log 2>&1 &
$PY run_eval3.py --wl wiki_t --extra --dr $DRS --v2 svdef,cross --out eval6 > $L/ev6_wiki_t.log 2>&1 &
wait; st S4
for wl in wiki_cf hotpot; do $PY run_eval3.py --wl $wl --extra --dr $DRS --v2 svdef,cross --out eval6 > $L/ev6_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t_q4 --extra --dr full:_rr,full:_q4 --v2 "" --out eval6 > $L/ev6_q4.log 2>&1 &
wait; st S5
for wl in wiki_t wiki_cf hotpot; do $PY run_eval3.py --wl $wl --dr full:_rr1,full:_rr2 --v2 "" --skip_nli --skip_margin --out eval6s > $L/ev6s_$wl.log 2>&1 & done
$PY run_eval3.py --wl wiki_t --dr full:_rrx --v2 "" --skip_nli --skip_margin --out eval6x > $L/ev6x.log 2>&1 &
wait
for wl in wiki_t wiki_cf hotpot; do DR_INT8=1 $PY run_eval3.py --wl $wl --dr full:_rr --v2 "" --skip_nli --skip_margin --out eval6q8 > $L/ev6q8_$wl.log 2>&1 & done
for wl in wiki_t wiki_cf; do EQUIV=strict $PY run_eval3.py --wl $wl --dr full:_rr --v2 cross --out eval6strict > $L/ev6strict_$wl.log 2>&1 & done
wait; st S6
$PY bench3.py --wl wiki_cf > $L/bench3_v6.log 2>&1
st V6DONE
