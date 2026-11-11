"""Extract the articles of the W1 (MQuAKE-T) entities from two REAL English Wikipedia dumps (2022-03-01, 2023-11-01)
and write them in the revision format consumed by build_wikirev.py:
  revs.json {entity: {"title", "snaps": {date: {"revid", "ts", "paras": [first 10 paragraphs >= 80 chars]}}}}
Titles are matched exactly, then case-insensitively."""
import glob, json, sys, re
import pyarrow.parquet as pq

ROOT = "/data/yaokang/evcache/data"
DUMPS = {"2022-03-01": f"{ROOT}/wikidumps/20220301", "2023-11-01": f"{ROOT}/wikidumps/20231101"}
ents = set()
for s in ["train", "cal", "test"]:
    for c in json.load(open(f"{ROOT}/wiki_t/split_{s}.json"))["cases"]:
        for t in c["triples"]:
            ents.add(t[0]); ents.add(t[2])
low = {e.lower(): e for e in ents}
MAXP = 10


def paras(text):
    out = []
    for p in text.split("\n"):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 80:
            out.append(p)
        if len(out) >= MAXP:
            break
    return out


found = {d: {} for d in DUMPS}
for d, path in DUMPS.items():
    for f in sorted(glob.glob(f"{path}/*.parquet")):
        t = pq.read_table(f, columns=["id", "title", "text"])
        for i, ti, tx in zip(t["id"].to_pylist(), t["title"].to_pylist(), t["text"].to_pylist()):
            e = ti if ti in ents else low.get(ti.lower())
            if e is not None and (e not in found[d] or ti == e):
                found[d][e] = {"revid": i, "ts": d, "paras": paras(tx), "title": ti}
        print(d, f, len(found[d]), flush=True)
out = {}
for e in ents:
    if all(e in found[d] for d in DUMPS) and found["2022-03-01"][e]["paras"]:
        out[e] = {"title": found["2022-03-01"][e]["title"], "snaps": {d: found[d][e] for d in DUMPS}}
json.dump(out, open(f"{ROOT}/wiki_rev/revs.json", "w"))
print("entities", len(ents), "matched in both dumps", len(out))
