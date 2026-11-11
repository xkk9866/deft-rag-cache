#!/bin/bash
# Round-7 queue, GPU 6 only. LLM jobs strictly sequential (GPU 6 is shared; vLLM's memory fraction counts other
# processes, so it is computed from current usage). ANN analysis runs alongside (CPU + tiny GPU footprint).
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache
st(){ echo "STAGE $1 $(date)" >> $L/stages_v19.log; }
mf(){ used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6); python3 -c "print(min(0.97, ($used + 9800) / 24564))"; }   # vLLM budget = current usage + 8.3 GB reader + 1.5 GB encoder (counted by vLLM)
export LLM_SEQS=8 LLM_MAXLEN=3072 LLM_EAGER=1 LLM_BTOK=2048
waitmem(){ while [ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6) -gt 14000 ]; do sleep 60; done; sleep 20; }   # wait for >= 10 GB free (GPU 6 is shared)
st v19_start
( for w in ; do $PY ann_reach.py --wl $w > $L/ann_$w.log 2>&1; done; st ANN_DONE ) &

# ---- W4: real Wikipedia revisions (build fixed: change test on paragraph text, not page id)
$PY build_wikirev.py > $L/build_w4.log 2>&1 || st W4_FAIL_build
for sp in test cal train; do waitmem; LLM_MEM=$(mf) $PY oracle.py --wl wiki_rev --split $sp --B 20 > $L/oracle_w4_$sp.log 2>&1 || st W4_FAIL_oracle_$sp; done
waitmem; LLM_MEM=$(mf) $PY label.py --wl wiki_rev > $L/label_w4.log 2>&1 || st W4_FAIL_label
st w4_oracle

# ---- speculative certification (reader-native, one teacher-forced pass)
for w in wiki_cf wiki_t hotpot wiki_rev; do waitmem; LLM_MEM=$(mf) $PY spec_cert.py --wl $w > $L/spec_$w.log 2>&1 || st SPEC_FAIL_$w; st spec_$w; done
st SPEC_DONE

# ---- W4 evaluation, re-distillation, audit, ANN
$PY run_eval.py --wl wiki_rev --models "" > $L/ev_w4_base.log 2>&1 || st W4_FAIL_base
DR_SETLEN=1024 $PY deltar.py --model setcross1 --epochs 2 --bs 8 --wls wiki_rev --init dr_setcross1_m3 --backbone $(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*) --ckpt --tag _wr > $L/dr_setcross1_wr.log 2>&1 || st W4_FAIL_distil
$PY run_eval3.py --wl wiki_rev --extra --dr setcross1:_m3,setcross1:_wr,full:_rr --v2 cross --out eval9 > $L/ev9_w4.log 2>&1 || st W4_FAIL_eval
$PY run_audit.py --wl wiki_rev --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3,set-cross1_wr > $L/audit_wiki_rev.log 2>&1 || st W4_FAIL_audit
$PY ann_reach.py --wl wiki_rev > $L/ann_wiki_rev.log 2>&1
st W4_DONE

# ---- closed-loop replay with three risk measures (all workloads)
: # W1-W3 closed loop runs in run_closed_g6.sh
$PY run_closed.py --wl wiki_rev --scorers margin,nli,pair-cross,DR_rr,set-cross1_m3,set-cross1_wr > $L/closed_wiki_rev.log 2>&1 || st CLOSED_FAIL_wiki_rev
wait
st V19_ALLDONE
