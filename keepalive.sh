#!/bin/bash
# restarts the queue scheduler if it is not running (stop with: touch /data/yaokang/evcache/keepalive.stop)
cd /data/yaokang/evcache
while true; do
  [ -f keepalive.stop ] && { echo "keepalive stop $(date)"; exit 0; }
  if ! pgrep -f 'code/qsched.py' >/dev/null; then
    [ -f qsched.stop ] && { sleep 60; continue; }
    echo "restarting qsched $(date)"
    QS_MAX=${QS_MAX:-3} QS_RETRIES=${QS_RETRIES:-3} QS_GPUS=${QS_GPUS:-0,1,2,3} setsid nohup /home/yaokang/research/qrag/.venv/bin/python code/qsched.py queue.txt >> logs/qsched.log 2>&1 < /dev/null &
  fi
  sleep 60
done
