from simulate import *
O = Oracle("wiki_t", "cal")
stream = zipf_stream(O.cids, O.nB, len(O.cids) // 2)
for p in ["none", "provenance", "trace_exact", "entity", "eager_all"]:
    t = time.time(); r = replay(O, p, stream); print(p, {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, f"{time.time()-t:.1f}s", flush=True)
t = time.time(); r = replay(O, "verify_on_hit", stream, verify=Verifier(O)); print("verify", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, f"{time.time()-t:.1f}s", flush=True)
sc = Scorer(O, "margin")
ds = defeat_scores(O, sc); print("defeat scores", len(ds), np.percentile(ds, [0, 5, 50, 95]))
lam = conformal_threshold(ds, 0.05); print("lam", lam)
r = replay(O, "learned", stream, scorer=sc, lam=lam); print("margin@0.05", {k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})
print(json.dumps(kind_breakdown(O, None, 0.0), indent=0))
