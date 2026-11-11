#!/bin/bash
# Dynamic GPU scheduler for the shared server.  Allowed: GPU 1 (main) always; any other GPU only if it has no process
# of another yaokang project (our own jobs run with cwd under /data/yaokang/evcache).  A job waits until an allowed GPU
# has `need` MiB free; picking + start-up happen under a lock so that concurrent lanes do not race for the same memory.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /data/yaokang/evcache/code
PY=/home/yaokang/research/qrag/.venv/bin/python; L=../logs; R=/data/yaokang/evcache/runs
st(){ echo "STAGE $1 $(date)" >> $L/stages_v20.log; }
M3=$(ls -d /home/yaokang/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots/*)

foreign(){   # 1 if GPU $1 runs a yaokang process of another project
  local uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $1)
  for p in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | grep "$uuid" | cut -d, -f2); do
    if [ "$(ps -o user= -p $p 2>/dev/null)" = "yaokang" ] && ! readlink /proc/$p/cwd 2>/dev/null | grep -q evcache; then echo 1; return; fi
  done; echo 0; }
pick(){      # print an allowed GPU with >= $1 MiB free (GPU 1 preferred), or nothing
  for g in 1 4 6 7 5 0 2 3; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
    [ "$free" -ge "$1" ] || continue
    [ "$g" = "1" ] || [ "$(foreign $g)" = "0" ] || continue
    echo $g; return
  done; }
# job <need MiB> <name> <cmd...> : run with a picked GPU; LLM jobs get LLM_MEM sized to that GPU
job(){
  local need=$1 name=$2; shift 2
  local g=""
  exec 9>/tmp/evcache_sched.lock
  while true; do flock 9; g=$(pick $need); [ -n "$g" ] && break; flock -u 9; sleep 60; done   # never wait while holding the lock
  local used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  export CUDA_VISIBLE_DEVICES=$g LLM_MEM=$(python3 -c "print(min(0.95, ($used + $need) / 24564))")
  echo "$(date +%T) $name -> GPU $g (free>=$need)" >> $L/sched_v20.log
  "$@" > $L/$name.log 2>&1 &
  local pid=$!
  sleep 90; flock -u 9
  wait $pid && st "ok_$name" || st "FAIL_$name"
}
