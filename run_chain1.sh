#!/bin/bash
cd /data/yaokang/evcache/code
for st in oracle_t hotpot oracle_cf label; do bash run_stage.sh $st; done
