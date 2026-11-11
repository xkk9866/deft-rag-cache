"""W4 (wiki_rev): an evolving corpus driven by REAL Wikipedia revision history.

Documents are paragraphs of the actual Wikipedia articles of every entity in the MQuAKE-T questions in the
2022-03-01 English Wikipedia dump.  The update stream is the real paragraph-level diff to the 2023-11-01 dump
(difflib alignment of paragraph lists): a changed paragraph is a `modify`, a new one an `insert`, a vanished one a
`delete`.  Articles are revised in random order, all paragraph edits of an article consecutively.  No text is generated.
Background: the 2017 Wikipedia paragraphs of W1.  Questions: the W1 (MQuAKE-T) cached questions and splits.
"""
import argparse, difflib, collections, random
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--revs", default=f"{DATA}/wiki_rev/revs.json")
args = ap.parse_args()
OUT = f"{DATA}/wiki_rev"
os.makedirs(OUT, exist_ok=True)
revs = {k: v for k, v in jload(args.revs).items() if v}
SN = ["2022-03-01", "2023-11-01"]
by_title = {}
for e, v in revs.items():
    by_title.setdefault(v["title"], v)
print("entities", len(revs), "articles", len(by_title))


def txt(title, snap, p):
    return f"{title} (Wikipedia, {snap['ts'][:10]}): {p}"


corpus = {k: v for k, v in jload(f"{DATA}/wiki_t/corpus.json").items() if k.startswith("B:")}
events_of = collections.defaultdict(list)          # title -> [(ts, event)]
nid = collections.Counter()
for title, v in by_title.items():
    s0 = v["snaps"][SN[0]]
    ids = []
    for p in s0["paras"]:
        d = f"W:{title}#{nid[title]}"; nid[title] += 1
        corpus[d] = txt(title, s0, p); ids.append(d)
    prev = s0["paras"]
    for sn in SN[1:]:
        cur = v["snaps"][sn]
        if cur["paras"] == prev:
            continue
        new_ids = []
        for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, prev, cur["paras"], autojunk=False).get_opcodes():
            if op == "equal":
                new_ids += ids[i1:i2]; continue
            m = min(i2 - i1, j2 - j1) if op == "replace" else 0
            for a in range(m):
                d = ids[i1 + a]
                events_of[title].append((cur["ts"], {"op": "modify", "kind": "rev_modify", "doc": d, "text": txt(title, cur, cur["paras"][j1 + a])}))
                new_ids.append(d)
            for a in range(i1 + m, i2):
                events_of[title].append((cur["ts"], {"op": "delete", "kind": "rev_delete", "doc": ids[a]}))
            for a in range(j1 + m, j2):
                d = f"W:{title}#{nid[title]}"; nid[title] += 1
                events_of[title].append((cur["ts"], {"op": "insert", "kind": "rev_insert", "doc": d, "text": txt(title, cur, cur["paras"][a])}))
                new_ids.append(d)
        ids, prev = new_ids, cur["paras"]
jdump(corpus, f"{OUT}/corpus.json")
print("corpus", len(corpus), "events", sum(len(x) for x in events_of.values()))

for split in ["train", "cal", "test"]:
    sp = jload(f"{DATA}/wiki_t/split_{split}.json")
    ents = {t[0] for c in sp["cases"] for t in c["triples"]} | {t[2] for c in sp["cases"] for t in c["triples"]}
    titles = {revs[e]["title"] for e in ents if e in revs}
    order = sorted(titles); random.Random(0).shuffle(order)       # articles revised in random order
    ev = [e for t in order for _, e in events_of.get(t, [])]
    jdump({"cases": sp["cases"], "events": ev}, f"{OUT}/split_{split}.json")
    print(split, "cases", len(sp["cases"]), "events", len(ev), collections.Counter(e["kind"] for e in ev))
