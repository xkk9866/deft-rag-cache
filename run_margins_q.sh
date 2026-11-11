#!/bin/bash
# margins.py for a workload replayed with another reader, on a shared GPU (vLLM engine; LLM_MEM via with_llm.sh).
# usage: run_margins_q.sh <wl> <llm_path> <need_MB> [splits=cal,test]
wl=$1; llm=$2; need=${3:-8800}; splits=${4:-cal,test}
export LLM_PATH=$llm
jobs=$(python3 -c "print(','.join('$wl:'+s for s in '$splits'.split(',')))")
synth=0; case ",$splits," in *,train,*) synth=3;; esac
exec bash "$(dirname "$0")/with_llm.sh" $need /home/yaokang/research/qrag/.venv/bin/python margins.py --jobs $jobs --synth $synth
