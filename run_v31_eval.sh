#!/bin/bash
# round 9 evaluation lane (GPU 6 only): re-split risk/saving for every new reader, request-level closed loops, transfer
export CUDA_VISIBLE_DEVICES=6 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache/runs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v31.log; }
waitm(){ while [ ! -f $R/models/dr_$1$2.json ]; do sleep 60; done; }
ev(){ $PY run_eval3.py --wl $3 --dr $1:$2 --v2 "" --skip_nli --skip_margin --out eval30_$1$2 > $L/ev30_$1$2_$3.log 2>&1 && st ok_ev_$1$2_$3 || st FAIL_ev_$1$2_$3; }
evm(){ waitm $1 $2; ev $1 $2 wiki_t & ev $1 $2 wiki_cf; wait; ev $1 $2 hotpot & ev $1 $2 wiki_rev; wait; }
cl(){ $PY run_closed2.py --wl $1 "${@:2}" > $L/cl2_$1_${OUTN}.log 2>&1 && st ok_cl2_$1_${OUTN} || st FAIL_cl2_$1_${OUTN}; }
evm mtr _s0
ev mtr _s0 wiki_t_q4 & ev mtr _s0 wiki_t_ll; wait
evm setcross1 _aug
for w in wiki_t wiki_cf hotpot wiki_rev; do
  OUTN=main cl $w --scorers mtr:_s0,setcross1:_aug,setcross1:_m3,nli,margin,uniform,oracle,mtr:_s0:raw --noshare --dump mtr:_s0 --out main
done
for w in wiki_t wiki_cf hotpot wiki_rev; do
  OUTN=gamma cl $w --scorers mtr:_s0 --alphas 0.01 --rhos 0.05 --gammas 0.003,0.03 --out gamma
done
evm mtr0 _s0
evm mtr _s1
evm mtr _s2
evm mtrE _s0
for w in wiki_t wiki_cf hotpot wiki_rev; do
  OUTN=abl cl $w --scorers mtr0:_s0,mtr:_s1,mtr:_s2,mtrE:_s0 --alphas 0.01 --rhos 0.05 --out abl
done
EQUIV=strict $PY run_eval3.py --wl wiki_t --dr mtr:_s0 --v2 "" --skip_nli --skip_margin --out eval30strict_mtr > $L/ev30strict_t.log 2>&1
EQUIV=strict $PY run_eval3.py --wl wiki_cf --dr mtr:_s0 --v2 "" --skip_nli --skip_margin --out eval30strict_mtr > $L/ev30strict_cf.log 2>&1
st EVAL_ALLDONE
