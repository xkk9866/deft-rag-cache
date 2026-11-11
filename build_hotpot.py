"""W3 (hotpot): HotpotQA distractor paragraphs as an evolving Wikipedia corpus.

Answer-changing insertions are LLM-written dated update notices that revise the answer-bearing fact
(old paragraphs stay untouched -- 'the citation did not change, the answer did').  Benign insertions:
restatements, same-entity news, unrelated paragraphs; plus in-place modifications and deletions.
"""
import argparse, random, collections, re
import pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--n_q", type=int, default=2000)
ap.add_argument("--n_bg", type=int, default=20000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
rng = random.Random(args.seed)
OUT = f"{DATA}/hotpot"

val = pd.read_parquet(f"{DATA}/hotpot/distractor/validation-00000-of-00001.parquet")
tr = pd.read_parquet(f"{DATA}/hotpot/distractor/train-00000-of-00002.parquet")
rows = [r for r in val.itertuples() if r.answer.lower() not in ("yes", "no")]
rng.shuffle(rows)
rows = rows[: args.n_q]

def para_of(ctx):
    return {t: t + ": " + "".join(ss).strip() for t, ss in zip(ctx["title"], ctx["sentences"])}

corpus, cases = {}, []
for r in rows:
    P = para_of(r.context)
    for t, p in P.items():
        corpus.setdefault("P:" + t, "(2017) " + p)
    sf = list(zip(r.supporting_facts["title"], r.supporting_facts["sent_id"]))
    sents = {t: list(ss) for t, ss in zip(r.context["title"], r.context["sentences"])}
    gold = []
    for t, i in sf:
        if t in sents and i < len(sents[t]):
            gold.append((t, sents[t][i].strip()))
    ans_title = next((t for t, s in gold if r.answer.lower() in s.lower()), gold[-1][0] if gold else None)
    cases.append({"cid": r.id, "q": r.question, "answer": r.answer, "answer_alias": [r.answer], "type": r.type,
                  "gold": gold, "ans_title": ans_title, "titles": sorted({t for t, _ in gold})})
used = {k[2:] for k in corpus}
filler = []
for ctx in tr["context"]:
    for t, ss in zip(ctx["title"], ctx["sentences"]):
        if t not in used:
            used.add(t); filler.append((t, t + ": " + "".join(ss).strip()))
rng.shuffle(filler)
for t, p in filler[: args.n_bg]:
    corpus["B:" + t] = "(2017) " + p
held = filler[args.n_bg: args.n_bg + 20000]
print("cases", len(cases), "corpus", len(corpus))

llm = LLMWrap(mem=0.55)
UPD = ("Question: {q}\nCurrent answer: {a}\nSupporting fact: {f}\n\nWrite ONE short news sentence dated 2024 reporting that this "
       "fact has changed (or has been officially corrected), so that the answer to the question would now be a different, "
       "plausible value of the same type. Name the entities explicitly. Then on a new line write 'NEW: <new answer>'.")
RESTATE = ("Rewrite this fact as one short news sentence published in 2024 that confirms it is still true. Keep every name "
           "exactly and do not add other facts.\nFact: {f}\nSentence:")
SIBL = ("Write one short news sentence published in 2024 about {t} that mentions {t} by name but says nothing about the "
        "following fact (do not mention or contradict it): {f}\nSentence:")
PARA = "Paraphrase this paragraph, keeping every fact and name unchanged:\n{p}\nParaphrase:"

idx = list(range(len(cases))); rng.shuffle(idx)
n = len(cases)
split_of = {}
for j, i in enumerate(idx):
    split_of[cases[i]["cid"]] = "train" if j < 0.3 * n else ("cal" if j < 0.45 * n else "test")

for split in ["train", "cal", "test"]:
    sc = [c for c in cases if split_of[c["cid"]] == split and c["gold"]]
    ev = []
    upd_c = sc[: int(0.45 * len(sc))]
    rest = sc[int(0.45 * len(sc)):]
    fact = lambda c: " ".join(s for t, s in c["gold"] if t == c["ans_title"]) or c["gold"][-1][1]
    outs = llm.chat([UPD.format(q=c["q"], a=c["answer"], f=fact(c)) for c in upd_c], max_tokens=96)
    news, newans = [], []
    for c, o in zip(upd_c, outs):
        lines = [l.strip() for l in o.split("\n") if l.strip()]
        sent = next((l for l in lines if not l.upper().startswith("NEW:")), "")
        na = next((l[4:].strip() for l in lines if l.upper().startswith("NEW:")), "")
        news.append(sent.strip('"')); newans.append(na)
    dups = llm.chat([f"Paraphrase this news sentence, keeping all names and the year:\n{s}\nParaphrase:" for s in news], max_tokens=96)
    for c, s, d2, na in zip(upd_c, news, dups, newans):
        if not s or not na:
            continue
        c["new_answer"] = na; c["new_answer_alias"] = [na]
        ev.append({"op": "insert", "kind": "update", "doc": "U:" + c["cid"], "text": "(2024) " + s, "cid": c["cid"]})
        ev.append({"op": "insert", "kind": "update_dup", "doc": "U2:" + c["cid"], "text": "(2024) " + d2.split("\n")[0].strip('"'), "cid": c["cid"]})
    nI = len(upd_c)
    rs = rng.sample(sc, min(len(sc), int(1.5 * nI)))
    outs = llm.chat([RESTATE.format(f=fact(c)) for c in rs], max_tokens=80)
    for c, o in zip(rs, outs):
        ev.append({"op": "insert", "kind": "restate", "doc": "R:" + c["cid"], "text": "(2024) " + o.split("\n")[0].strip('"')})
    sb = rng.sample(sc, min(len(sc), int(1.5 * nI)))
    outs = llm.chat([SIBL.format(t=c["ans_title"], f=fact(c)) for c in sb], max_tokens=80)
    for c, o in zip(sb, outs):
        ev.append({"op": "insert", "kind": "sibling", "doc": "S:" + c["cid"], "text": "(2024) " + o.split("\n")[0].strip('"')})
    for t, p in rng.sample(held, 3 * nI):
        ev.append({"op": "insert", "kind": "random", "doc": "B:" + t, "text": "(2024) " + p})
    # in-place modification of cited paragraphs (benign paraphrase / value change) on non-updated questions
    mb = rest[: nI // 3]
    outs = llm.chat([PARA.format(p=corpus["P:" + c["ans_title"]][7:]) for c in mb], max_tokens=256)
    for c, o in zip(mb, outs):
        ev.append({"op": "modify", "kind": "modify_benign", "doc": "P:" + c["ans_title"], "text": "(2017) " + o.strip()})
    mv = rest[nI // 3: 2 * (nI // 3)]
    others = [c["answer"] for c in cases]
    for c in mv:
        p = corpus["P:" + c["ans_title"]]
        if c["answer"] not in p:
            continue
        alt = rng.choice([a for a in others if a != c["answer"] and len(a.split()) == len(c["answer"].split())] or others)
        ev.append({"op": "modify", "kind": "modify_value", "doc": "P:" + c["ans_title"], "text": p.replace(c["answer"], alt)})
    for c in rest[2 * (nI // 3): 2 * (nI // 3) + nI // 3]:
        ev.append({"op": "delete", "kind": "delete_fact", "doc": "P:" + c["ans_title"]})
    for t in rng.sample([k for k in corpus if k.startswith("B:")], nI // 3):
        ev.append({"op": "delete", "kind": "delete_bg", "doc": t})
    rng.shuffle(ev)
    pos = {e["doc"]: i for i, e in enumerate(ev)}
    for e in list(ev):
        if e["kind"] == "update_dup":
            i, j = pos["U:" + e["cid"]], pos[e["doc"]]
            if i > j:
                ev[i], ev[j] = ev[j], ev[i]; pos[ev[i]["doc"]], pos[ev[j]["doc"]] = i, j
    seen, ev2 = set(), []
    for e in ev:
        if e["doc"] not in seen:
            seen.add(e["doc"]); ev2.append(e)
    recs = [{"cid": c["cid"], "q": c["q"], "answer": c["answer"], "answer_alias": c["answer_alias"],
             "new_answer": c.get("new_answer", c["answer"]), "new_answer_alias": c.get("new_answer_alias", c["answer_alias"]),
             "type": c["type"]} for c in sc]
    jdump({"cases": recs, "events": ev2}, f"{OUT}/split_{split}.json")
    print(split, len(recs), len(ev2), collections.Counter(e["kind"] for e in ev2), flush=True)
jdump(corpus, f"{OUT}/corpus.json")
print("corpus", len(corpus))
