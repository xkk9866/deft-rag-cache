"""one-off patch applying the v2 changes (fragility keys + batch-level CRC)"""
p = 'deft.py'; s = open(p, encoding='utf-8').read()
s = s.replace('''        self.t_v = nn.Parameter(torch.tensor(math.log(20.0)))
''', '''        self.t_v = nn.Parameter(torch.tensor(math.log(20.0)))
        self.frag = kind.startswith("svdef")
        if self.frag:              # fragility head: how weakly the hop answer is supported by its own evidence
            self.Pf = nn.Linear(H, 1)
''', 1)
s = s.replace('''    def key(self, subqs, answers):
        if self.kind == "bienc":''', '''    def key(self, subqs, answers, evid=None):
        if self.kind == "bienc":''')
s = s.replace('''        return {"s": F.normalize(self.Ps(hs), dim=-1), "v": F.normalize(self.Pv(hv), dim=-1)}

    def probe''', '''        K = {"s": F.normalize(self.Ps(hs), dim=-1), "v": F.normalize(self.Pv(hv), dim=-1)}
        if self.frag and evid is not None:
            hf = self.cls_pair([f"Question: {q} Answer: {a}" for q, a in zip(subqs, answers)], list(evid), 384)
            K["phi"] = self.Pf(hf).squeeze(-1)
        return K

    def cls_pair(self, left, right, max_len):
        dev = next(self.parameters()).device
        b = self.tok(left, right, padding=True, truncation="only_second", max_length=max_len, return_tensors="pt").to(dev)
        return self.enc(**b).last_hidden_state[:, 0]

    def probe''')
s = s.replace('''        return self.a * ss - self.b * vv + self.c, self.a_s * ss + self.c_s''', '''        phi = K["phi"] if "phi" in K else 0.0
        return self.a * ss - self.b * vv + self.c + phi, self.a_s * ss + self.c_s''')
s = s.replace('''    def key_vectors(self, subqs, answers, bs=256):
        out = []
        for i in range(0, len(subqs), bs):
            K = self.key(subqs[i:i + bs], answers[i:i + bs])''', '''    def key_vectors(self, subqs, answers, evid=None, bs=128):
        out = []
        for i in range(0, len(subqs), bs):
            K = self.key(subqs[i:i + bs], answers[i:i + bs], evid[i:i + bs] if evid is not None else None)''')
s = s.replace('''                out.append(torch.cat([self.a * K["s"], -self.b * K["v"], self.c.expand(len(K["s"]), 1)], 1))''', '''                bias = self.c.expand(len(K["s"])) + (K["phi"] if "phi" in K else 0.0)
                out.append(torch.cat([self.a * K["s"], -self.b * K["v"], bias.reshape(-1, 1)], 1))''')
s = s.replace('''                    K = model.key([r["subq"] for r in B], [r["ans"] for r in B])
                    P = model.probe''', '''                    K = model.key([r["subq"] for r in B], [r["ans"] for r in B], [r.get("ev", "") for r in B])
                    P = model.probe''')
s = s.replace('''                    if args.model == "svde":''', '''                    if args.model in ("svde", "svdef"):''')
s = s.replace('''            rs = jload(p)
            for r in rs:''', '''            rs = jload(p)
            R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
            def vb(cid, b):
                v = None
                for x, r_ in R["versions"][cid]:
                    if x < b:
                        v = r_
                return v
            for r in rs:
                hop = vb(r["cid"], r["b"])["trace"][r["h"]]
                r["ev"] = " ".join(R["texts"][x] for x in hop["rows"])''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'simulate.py'; s = open(p, encoding='utf-8').read()
s = s.replace('''            K = self.model.key_vectors(subqs, ans)''', '''            ev = [" ".join(self.O.texts[x] for x in self.O.trace(c, v)[h]["rows"]) for c, v, h in miss]
            K = self.model.key_vectors(subqs, ans, ev if getattr(self.model, "frag", False) else None)''')
s = s.replace('''        if self.kind in ("svde", "svde_noaux", "bienc"):''', '''        if self.kind in ("svde", "svde_noaux", "bienc", "svdef", "svdef_noaux"):''')
s = s.replace('''def conformal_threshold(scores, alpha):''', '''def defeat_scores_batched(O, scorer, use_reach=True, qsim=None):
    """per batch: scores of the entries the batch defeats (only batches with at least one defeat)."""
    out = []
    for b in range(1, O.nB + 1):
        defeated = [cid for cid, c in O.logs[b - 1]["changed"].items() if c]
        if not defeated:
            continue
        sub = {cid: O.cur_vidx(cid, b - 1) for cid in defeated}
        s = entry_scores(O, sub, b, scorer, use_reach, qsim)
        out.append(np.array([s[c] for c in defeated]))
    return out


def crc_threshold(batches, alpha):
    """Conformal risk control with batches as exchangeable units; loss = missed fraction of the batch's
    defeated entries (bounded by 1). Largest lambda with n/(n+1) R_n(lambda) + 1/(n+1) <= alpha."""
    n = len(batches)
    if n == 0 or 1.0 / (n + 1) > alpha:
        return -np.inf
    cand = np.unique(np.concatenate([x[np.isfinite(x)] for x in batches] + [np.array([-np.inf])]))
    best = -np.inf
    for lam in cand:
        risk = np.mean([np.mean(x < lam) for x in batches])
        if n / (n + 1) * risk + 1.0 / (n + 1) <= alpha:
            best = lam
        else:
            break
    return float(best)


def conformal_threshold(scores, alpha):''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)

p = 'run_eval.py'; s = open(p, encoding='utf-8').read()
s = s.replace('''    cs = defeat_scores(Oc, sc_c, use_reach, qsim_c if qsim else None)''', '''    cb = defeat_scores_batched(Oc, sc_c, use_reach, qsim_c if qsim else None)
    cs = np.concatenate(cb) if cb else np.zeros(0)''')
s = s.replace('''        lam = conformal_threshold(cs, a)''', '''        lam = crc_threshold(cb, a)''')
s = s.replace('''    ts = defeat_scores(Ot, sc_t, use_reach, qsim_t if qsim else None)''', '''    tb = defeat_scores_batched(Ot, sc_t, use_reach, qsim_t if qsim else None)
    ts = np.concatenate(tb) if tb else np.zeros(0)''')
s = s.replace('''        res.update({"lambda": lam,''', '''        res.update({"miss_batch": float(np.mean([np.mean(x < lam) for x in tb])) if tb else 0.0, "n_cal_batches": len(cb), "lambda": lam,''')
s = s.replace('''ap.add_argument("--models", default="svde,svde_noaux,bienc,cross")''', '''ap.add_argument("--models", default="svdef,svde,svdef_noaux,bienc,cross")''')
s = s.replace('''    alphas = (args.alpha, 0.01, 0.02, 0.1, 0.2) if m == "svde" else (args.alpha,)
    calibrated({"svde": "DEFT", "svde_noaux": "DEFT-noaux", "bienc": "DEFT-bienc", "cross": "DEFT-cross"}[m],''', '''    alphas = (args.alpha, 0.01, 0.02, 0.1, 0.2) if m == "svdef" else (args.alpha,)
    calibrated({"svdef": "DEFT", "svde": "DEFT-nofrag", "svdef_noaux": "DEFT-noaux", "bienc": "DEFT-bienc", "cross": "DEFT-cross"}[m],''')
s = s.replace('''    if m == "svde":
        calibrated("DEFT-noreach", lambda O, model=model: Scorer(O, "svde", model=model), use_reach=False, sweep=False)''', '''    if m == "svdef":
        calibrated("DEFT-noreach", lambda O, model=model: Scorer(O, "svdef", model=model), use_reach=False, sweep=False)''')
open(p, 'w', encoding='utf-8', newline='\n').write(s)
print("patched")
