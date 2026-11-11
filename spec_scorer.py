"""Batch-level scorer interface of the speculator for run_eval3 (Save@alpha under batch-level conformal risk control).
The entry score is the maximum over its touched hops of the speculator's defeat score for the cumulative edit
K* (the evidence the cached hop answer was produced on) -> K' (the exact current top-k)."""
import time
import numpy as np
import torch
from common import *
from common import equiv
from simulate import SetScorer, answer_rows
from mcert import MarginBook


class PseudoBook:
    """fallback when the reader's token ids were not logged for a replay (no margins.pkl): the cached tokens are
    reconstructed by tokenising the recorded hop answer followed by the end-of-turn token; trusted margins are unknown (0)."""

    def __init__(self, O, tok):
        self.S = {}
        eos = [i for i in {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|eot_id|>")} if isinstance(i, int) and i >= 0]
        for cid, vs in O.versions.items():
            for _, r in vs:
                for h in r["trace"]:
                    k = (h["subq"], tuple(h["rows"]))
                    if k not in self.S:
                        self.S[k] = tok(h["ans"], add_special_tokens=False)["input_ids"] + eos[:1]

    def tokens(self, q, rows):
        return self.S.get((q, tuple(rows)))

    def mu(self, q, rows):
        return 0.0


class SpecScorer(SetScorer):
    set_level = True

    def __init__(self, O, spec, mode="zs"):
        self.O, self.spec, self.mode = O, spec, mode
        self.memo, self.time = {}, 0.0
        self.MB = MarginBook(O.wl, O.split)
        if not self.MB.S or os.environ.get("SPEC_PSEUDO"):
            print("SpecScorer: reconstructing cached tokens from recorded answers for", O.wl, O.split, flush=True)
            self.MB = PseudoBook(O, spec.tok)
        self.ns = f"{O.wl}/{O.split}"

    def entry_scores(self, held, b):
        O = self.O
        t = time.time()
        sc = {cid: -np.inf for cid, v in held.items() if v is not None}
        hops = self.touched(held, b)
        need = [it for it in hops if (it, b) not in self.memo]
        if need:
            Q = torch.stack([O.qemb[O.trace(c, v)[h]["subq"]] for c, v, h in need])
            s = Q @ O.E.T
            s[:, ~O.alive[b]] = -1e4
            top = torch.topk(s, O.k, dim=1).indices.tolist()
            cur_items, tru_items, mus = [], [], []
            for (c, v, h), k2 in zip(need, top):
                hop = O.trace(c, v)[h]
                K = tuple(answer_rows(O, c, v, h))
                y = tuple(self.MB.tokens(hop["subq"], K) or ())
                cur_items.append((hop["subq"], tuple(k2), y)); tru_items.append((hop["subq"], K, y))
                mus.append(self.MB.mu(hop["subq"], K))
            if self.mode == "judge":                                    # same small model asked yes/no
                z = -np.array(self.spec.judge(O.texts, [(q, k2, O.trace(c, v)[h]["ans"]) for (c, v, h), (q, k2, y) in zip(need, cur_items)], ns=self.ns))
                for it, x in zip(need, z):
                    self.memo[(it, b)] = float(x)
                for c, v, h in hops:
                    sc[c] = max(sc[c], self.memo[((c, v, h), b)])
                self.time += time.time() - t
                return sc
            if self.mode == "reans":                                    # same small model re-answers; differs -> 1
                gen = self.spec.reanswer(O.texts, [(q, k2) for q, k2, y in cur_items], ns=self.ns)
                z = np.array([0.0 if equiv(g, O.trace(c, v)[h]["ans"]) else 1.0 for g, (c, v, h) in zip(gen, need)])
                for it, x in zip(need, z):
                    self.memo[(it, b)] = float(x)
                for c, v, h in hops:
                    sc[c] = max(sc[c], self.memo[((c, v, h), b)])
                self.time += time.time() - t
                return sc
            S2 = self.spec.stats(O.texts, cur_items, ns=self.ns)
            cur = np.array([s["m"] for s in S2])
            if self.mode == "zs":
                z = -cur
            else:
                S1 = self.spec.stats(O.texts, tru_items, ns=self.ns)
                tru = np.array([s["m"] for s in S1])
                if self.mode == "shift":
                    z = -(cur - tru)
                elif self.mode == "ll":
                    z = -np.array([b["ll"] - a_["ll"] for a_, b in zip(S1, S2)])
                else:                                               # did: trusted reader margin transported by the sibling's shift
                    z = -(np.log1p(np.maximum(np.array(mus), 0.0)) + cur - tru)
            for it, x in zip(need, z):
                self.memo[(it, b)] = float(x)
        for c, v, h in hops:
            sc[c] = max(sc[c], self.memo[((c, v, h), b)])
        self.time += time.time() - t
        return sc
