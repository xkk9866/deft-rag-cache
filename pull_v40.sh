#!/bin/bash
# collect round-11 results into a tarball: /data/yaokang/evcache/res11.tgz  (extract locally into results11/)
cd /data/yaokang/evcache
rm -rf res11 && mkdir -p res11
for w in wiki_t wiki_cf hotpot wiki_rev wiki_t_q4 wiki_t_ll; do
  mkdir -p res11/$w
  for f in main2 base2 spec2 spec3 gamma2 abl2 alt2 size2; do [ -f runs/$w/closed2/$f.json ] && cp runs/$w/closed2/$f.json res11/$w/closed2_$f.json; done
  for e in eval40 eval40p eval41 eval42 eval40b eval43 eval44 eval45 eval46; do [ -f runs/$w/$e/results.json ] && cp runs/$w/$e/results.json res11/$w/$e.json; done
  for b in spec_bench spec_bench17 spec_bench2; do [ -f runs/$w/$b.json ] && cp runs/$w/$b.json res11/$w/$b.json; done
  [ -f runs/$w/serve3/results.json ] && cp runs/$w/serve3/results.json res11/$w/serve3.json
  [ -f runs/$w/verify2.json ] && cp runs/$w/verify2.json res11/$w/verify2.json
done
cp runs/probe_*.json res11/ 2>/dev/null
for d in runs/models/spec_*; do [ -f $d/scale.json ] && mkdir -p res11/models/$(basename $d) && cp $d/scale.json res11/models/$(basename $d)/; done
tar czf res11.tgz res11 && ls -la res11.tgz && find res11 -type f | sort
