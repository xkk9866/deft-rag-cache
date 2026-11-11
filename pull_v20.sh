#!/bin/bash
# copy round-8 results from the server into results9/<wl>/ (and refresh results8 closed-loop files)
cd "Y:/xkk_workspace/workspace/ICDE/6"
S=/data/yaokang/evcache/runs
for w in wiki_t wiki_cf hotpot wiki_rev; do
  mkdir -p results9/$w
  for f in "closed/results.json:closed" "closed/cold.json:cold" "closed/cross.json:cross" "closed/closed_delta.json:closed_delta" \
           "eval11/results.json:eval11" "ann/results.json:ann" "serve/results.json:serve" "closed/sweep10.json:sweep10" "closed/sweep30.json:sweep30"; do
    src=${f%%:*}; dst=${f##*:}
    timeout 120 scp -q esr:$S/$w/$src results9/$w/$dst.json 2>/dev/null
  done
  [ -s results9/$w/closed.json ] && cp results9/$w/closed.json results8/$w/closed.json
  [ -s results9/$w/ann.json ] && cp results9/$w/ann.json results8/$w/ann.json
done
ls -la results9/*
