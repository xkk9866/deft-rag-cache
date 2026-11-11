"""Claim-level teacher signals: the reader reads every (sub-question, document) pair of the training evidence sets
alone (K and K' of every training example) and states which value the document asserts for the sub-question,
or 'unknown'.  These supervise the slot-conditioned reading r(d), v(d), g(d) of the Delta-Reader for *every*
document of an evidence set, not only for the inserted one."""
import argparse
from common import *
from rag import clean_ans
from deltar import edit_examples, pair_examples

READ = ("Read the document and answer the question using only this document. If the document does not state the answer, "
        "reply exactly 'unknown'. Reply with only the answer.\n\nDocument: {d}\n\nQuestion: {q}\nAnswer:")
ap = argparse.ArgumentParser()
ap.add_argument("--wls", default="wiki_cf,hotpot,wiki_t")
args = ap.parse_args()
todo, texts = {}, {}
for wl in args.wls.split(","):
    e1, T = edit_examples(wl); texts[wl] = T
    for e in e1 + pair_examples(wl, T):
        for d in set(e["K"]) | set(e["K2"]):
            todo[(wl, e["subq"], d[1])] = None
keys = sorted(todo)
print("unique (subq, doc) reads", len(keys), flush=True)
llm = LLMWrap(mem=0.6)
outs = llm.chat([READ.format(d=texts[wl][row], q=q) for wl, q, row in keys], max_tokens=24)
rec = [{"wl": wl, "subq": q, "row": row, "read": clean_ans(o)} for (wl, q, row), o in zip(keys, outs)]
for wl in args.wls.split(","):
    jdump([r for r in rec if r["wl"] == wl], f"{RUNS}/{wl}/train/reads.json")
print("relevant", sum(normalize(r["read"]) not in ("unknown", "") for r in rec) / len(rec), flush=True)
