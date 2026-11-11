"""print Save@alpha of every scorer in the eval files of results11 (usage: python show_ev.py ../results11 [wl ...])"""
import json, os, sys, glob
root = sys.argv[1] if len(sys.argv) > 1 else "../results11"
wls = sys.argv[2:] or ["wiki_t", "wiki_cf", "hotpot", "wiki_rev", "wiki_t_q4", "wiki_t_ll"]
for w in wls:
    for f in sorted(glob.glob(f"{root}/{w}/eval4*.json")):
        R = json.load(open(f))
        for k, v in R.get("split", {}).items():
            print(f"{w:10s} {os.path.basename(f):12s} {k:36s}", " ".join(f"{100 * v[a]['save_mean']:5.1f}(r{100 * v[a]['risk_mean']:4.1f})" for a in ("0.05", "0.1", "0.2") if a in v))
