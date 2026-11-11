"""Slot-Value Defeater Encoder (SVDE) and learned baselines.

SVDE: one transformer backbone, two shared projections.
    key(h)   = [ a * sigma(q_h) ; -b * nu(a_h) ; c ]       (stored in the defeater index)
    probe(d) = [ sigma~(d)      ;  nu~(d)      ; 1 ]
    S(h,d)   = <key, probe> = a<sigma,sigma~> - b<nu,nu~> + c
'same slot, different value'  ->  high;  'same slot, same value' or 'other slot'  ->  low.
Trained by counterfactual distillation from the LLM:
    L = BCE(S, y) + BCE(a_s<sigma,sigma~> + c_s, r) + InfoNCE(nu~(d), nu(v_d))
Variants (--model): svde | svde_noaux | bienc (unfactorised dual encoder) | cross (cross-encoder)
"""
import argparse, random, math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from common import *


class SVDE(nn.Module):
    def __init__(self, path, dim=256, kind="svde"):
        super().__init__()
        self.kind = kind
        self.tok = AutoTokenizer.from_pretrained(path)
        self.enc = AutoModel.from_pretrained(path)
        H = self.enc.config.hidden_size
        if kind == "cross":
            self.head = nn.Linear(H, 1)
        else:
            self.Ps = nn.Linear(H, dim)
            self.Pv = nn.Linear(H, dim) if kind.startswith("svde") else None
            self.Pk = nn.Linear(H, dim) if kind == "bienc" else None
        self.a = nn.Parameter(torch.tensor(8.0)); self.b = nn.Parameter(torch.tensor(8.0)); self.c = nn.Parameter(torch.tensor(-2.0))
        self.a_s = nn.Parameter(torch.tensor(8.0)); self.c_s = nn.Parameter(torch.tensor(-4.0))
        self.t_v = nn.Parameter(torch.tensor(math.log(20.0)))
        self.frag = kind.startswith("svdef")
        if self.frag:              # fragility head: how weakly the hop answer is supported by its own evidence
            self.Pf = nn.Linear(H, 1)

    def cls(self, texts, max_len):
        dev = next(self.parameters()).device
        b = self.tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        return self.enc(**b).last_hidden_state[:, 0]

    # ---- dual-encoder side representations
    def key(self, subqs, answers, evid=None):
        if self.kind == "bienc":
            h = self.cls([QPREFIX + q + " Current answer: " + a for q, a in zip(subqs, answers)], 96)
            return {"f": F.normalize(self.Pk(h), dim=-1)}
        hs = self.cls([QPREFIX + q for q in subqs], 64)
        hv = self.cls(list(answers), 24)
        K = {"s": F.normalize(self.Ps(hs), dim=-1), "v": F.normalize(self.Pv(hv), dim=-1)}
        if self.frag and evid is not None:
            hf = self.cls_pair([f"Question: {q} Answer: {a}" for q, a in zip(subqs, answers)], list(evid), 384)
            K["phi"] = self.Pf(hf).squeeze(-1)
        return K

    def cls_pair(self, left, right, max_len):
        dev = next(self.parameters()).device
        b = self.tok(left, right, padding=True, truncation="only_second", max_length=max_len, return_tensors="pt").to(dev)
        return self.enc(**b).last_hidden_state[:, 0]

    def probe(self, docs):
        h = self.cls(list(docs), 192)
        if self.kind == "bienc":
            return {"f": F.normalize(self.Ps(h), dim=-1)}
        return {"s": F.normalize(self.Ps(h), dim=-1), "v": F.normalize(self.Pv(h), dim=-1)}

    def pair_logits(self, K, P):
        """elementwise scores for aligned key/probe batches"""
        if self.kind == "bienc":
            return self.a * (K["f"] * P["f"]).sum(-1) + self.c, None
        ss = (K["s"] * P["s"]).sum(-1); vv = (K["v"] * P["v"]).sum(-1)
        phi = K["phi"] if "phi" in K else 0.0
        return self.a * ss - self.b * vv + self.c + phi, self.a_s * ss + self.c_s

    def cross_logits(self, subqs, answers, docs):
        dev = next(self.parameters()).device
        left = [f"Question: {q} Current answer: {a}" for q, a in zip(subqs, answers)]
        b = self.tok(left, list(docs), padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
        return self.head(self.enc(**b).last_hidden_state[:, 0]).squeeze(-1)

    # ---- index vectors (MIPS form)
    @torch.no_grad()
    def key_vectors(self, subqs, answers, evid=None, bs=128):
        out = []
        for i in range(0, len(subqs), bs):
            K = self.key(subqs[i:i + bs], answers[i:i + bs], evid[i:i + bs] if evid is not None else None)
            if self.kind == "bienc":
                out.append(torch.cat([self.a * K["f"], self.c.expand(len(K["f"]), 1)], 1))
            else:
                bias = self.c.expand(len(K["s"])) + (K["phi"] if "phi" in K else 0.0)
                out.append(torch.cat([self.a * K["s"], -self.b * K["v"], bias.reshape(-1, 1)], 1))
        return torch.cat(out).float()

    @torch.no_grad()
    def probe_vectors(self, docs, bs=128):
        out = []
        for i in range(0, len(docs), bs):
            P = self.probe(docs[i:i + bs])
            parts = [P["f"]] if self.kind == "bienc" else [P["s"], P["v"]]
            out.append(torch.cat(parts + [torch.ones(len(parts[0]), 1, device=parts[0].device)], 1))
        return torch.cat(out).float()


def load_pairs(wls):
    recs = []
    for wl in wls:
        p = f"{RUNS}/{wl}/train/pairs.json"
        if os.path.exists(p):
            rs = jload(p)
            R = pload(f"{RUNS}/{wl}/train/oracle.pkl")
            def vb(cid, b):
                v = None
                for x, r_ in R["versions"][cid]:
                    if x < b:
                        v = r_
                return v
            for r in rs:
                hop = vb(r["cid"], r["b"])["trace"][r["h"]]
                r["ev"] = " ".join(R["texts"][x] for x in hop["rows"])
                r["wl"] = wl
                if r["reach"]:
                    r["y"] = int(not equiv(r["cf_ans"], r["ans"]))
            recs += rs
            print(wl, len(rs), flush=True)
    return recs


def train(args):
    torch.manual_seed(0); random.seed(0)
    recs = load_pairs(args.wls.split(","))
    model = SVDE(bge_path(), kind=args.model).cuda()
    enc_params = list(model.enc.parameters())
    other = [p for n, p in model.named_parameters() if not n.startswith("enc.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": other, "lr": 1e-3}], weight_decay=0.01)
    pos = sum(r["y"] for r in recs); neg = len(recs) - pos
    pw = torch.tensor(neg / max(pos, 1), device="cuda")
    print("train pairs", len(recs), "pos", pos, "pos_weight", float(pw), flush=True)
    steps = args.epochs * math.ceil(len(recs) / args.bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.06 * steps)) * max(0.0, (steps - s) / steps))
    model.train(); st = 0
    for ep in range(args.epochs):
        random.shuffle(recs)
        tot = 0.0
        for i in range(0, len(recs), args.bs):
            B = recs[i:i + args.bs]
            y = torch.tensor([r["y"] for r in B], dtype=torch.float, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if args.model == "cross":
                    lg = model.cross_logits([r["subq"] for r in B], [r["ans"] for r in B], [r["doc"] for r in B])
                    loss = F.binary_cross_entropy_with_logits(lg.float(), y, pos_weight=pw)
                else:
                    K = model.key([r["subq"] for r in B], [r["ans"] for r in B], [r.get("ev", "") for r in B])
                    P = model.probe([r["doc"] for r in B])
                    lg, slg = model.pair_logits(K, P)
                    loss = F.binary_cross_entropy_with_logits(lg.float(), y, pos_weight=pw)
                    if args.model in ("svde", "svdef"):
                        rr = torch.tensor([r["r"] for r in B], dtype=torch.float, device="cuda")
                        loss = loss + F.binary_cross_entropy_with_logits(slg.float(), rr)
                        rel = [j for j, r in enumerate(B) if r["r"] and r["read"]]
                        if len(rel) > 1:
                            vt = model.key([B[j]["subq"] for j in rel], [B[j]["read"] for j in rel])["v"]
                            pv = P["v"][rel]
                            sim = (pv @ vt.T).float() * model.t_v.exp()
                            vals = [normalize(B[j]["read"]) for j in rel]
                            same = torch.tensor([[a == b for b in vals] for a in vals], device="cuda")
                            sim = sim.masked_fill(same & ~torch.eye(len(rel), dtype=torch.bool, device="cuda"), -1e4)
                            loss = loss + F.cross_entropy(sim, torch.arange(len(rel), device="cuda"))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); st += 1
            tot += float(loss)
        print(f"epoch {ep} loss {tot / math.ceil(len(recs) / args.bs):.4f}", flush=True)
    os.makedirs(f"{RUNS}/models", exist_ok=True)
    torch.save(model.state_dict(), f"{RUNS}/models/{args.model}{args.tag}.pt")
    print("saved", flush=True)


def load_model(kind, tag=""):
    m = SVDE(bge_path(), kind=kind).cuda().eval()
    m.load_state_dict(torch.load(f"{RUNS}/models/{kind}{tag}.pt", map_location="cuda"))
    return m


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="svde")
    ap.add_argument("--wls", default="wiki_cf,hotpot,wiki_t")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--tag", default="")
    train(ap.parse_args())
