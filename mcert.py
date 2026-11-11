"""Margin certificates of cached hop answers (helpers shared by training, replay and analysis).

A hop answer y = (y_1..y_n, <stop>) produced greedily on evidence K carries the decision margin
    mu(y; K) = min_i [ log p(y_i | x_K, y_<i) - max_{v != y_i} log p(v | x_K, y_<i) ]  >= 0,
read off the top-2 log-probabilities the reader computes anyway.  After an edit K -> K' the *signed* margin of the
cached tokens y on K' is
    m(y; K') = mu(y'; K')                         if greedy decoding on K' reproduces y (y' = y),
             = log p(y_i | ..) - log p(y'_i | ..) at the first position i where y' departs from y   (< 0),
and the hop is defeated (token-exactly) iff m(y; K') < 0.  When y_i is not among the reader's top-20 tokens at the
departure the value is only bounded above by the 20th log-probability (a censored observation).
"""
from common import *

MCAP = 20.0     # margins are measured in nats; log-probs below the top-20 floor are censored


def load_margins(wl, split):
    p = f"{RUNS}/{wl}/{split}/margins.pkl"
    return pload(p) if os.path.exists(p) else None


def mu(st):
    """trusted margin of the state's own greedy answer (min top-1 minus top-2 gap over all positions incl. stop)"""
    if st is None or len(st["lp"]) == 0:
        return 0.0
    lp = st["lp"].astype(np.float32)
    gap = lp[:, 0] - np.maximum(lp[:, 1], lp[:, 0] - MCAP)
    return float(np.clip(gap.min(), 0.0, MCAP))


def signed_margin(y_tok, st2):
    """(m, censored): signed margin of cached tokens y_tok on the post-edit state st2 (see module doc)."""
    t2 = st2["tok"]
    n = min(len(y_tok), len(t2))
    i = next((j for j in range(n) if y_tok[j] != t2[j]), None)
    if i is None:
        if len(y_tok) == len(t2):
            return mu(st2), False
        i = n                                           # one sequence stops (or continues) where the other does not
        if i >= len(st2["lp"]):
            return -1.0, True
    lp, ids = st2["lp"][i].astype(np.float32), st2["ids"][i]
    want = y_tok[i] if i < len(y_tok) else None
    hit = np.nonzero(ids == want)[0] if want is not None else []
    if len(hit):
        return float(max(lp[hit[0]] - lp[0], -MCAP)), False
    return float(max(lp[-1] - lp[0], -MCAP)), True     # y_i below the top-20: m <= lp_20 - lp_1


class MarginBook:
    """margins of every hop state of a (workload, split), keyed by (sub-question, evidence rows)"""

    def __init__(self, wl, split):
        M = load_margins(wl, split)
        self.S = M["states"] if M else {}
        self.synth = M["synth"] if M else []

    def has(self, q, rows):
        return (q, tuple(rows)) in self.S

    def mu(self, q, rows):
        return mu(self.S.get((q, tuple(rows))))

    def tokens(self, q, rows):
        st = self.S.get((q, tuple(rows)))
        return st["tok"] if st else None

    def post(self, q, rows, rows2):
        """signed post-edit margin of the answer generated on `rows` when the evidence becomes `rows2`"""
        a, b = self.S.get((q, tuple(rows))), self.S.get((q, tuple(rows2)))
        if a is None or b is None:
            return None, None
        return signed_margin(a["tok"], b)
