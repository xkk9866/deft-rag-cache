"""Remote, byte-range access to two real Wikipedia dumps on hf-mirror (no full download):
phase 1 reads only the `title` column of every row group and locates the W1 entities;
phase 2 reads the `text` column of the matching row groups only and writes revs.json for build_wikirev.py."""
import json, re, sys, time, os
import concurrent.futures as cf
import fsspec, pyarrow.parquet as pq

ROOT = "/data/yaokang/evcache/data"
M = "https://hf-mirror.com/datasets"
DUMPS = {"2022-03-01": f"{M}/legacy-datasets/wikipedia/resolve/main/data/20220301.en/train-{{:05d}}-of-00041.parquet",
         "2023-11-01": f"{M}/wikimedia/wikipedia/resolve/main/20231101.en/train-{{:05d}}-of-00041.parquet"}
ents = set()
for s in ["train", "cal", "test"]:
    for c in json.load(open(f"{ROOT}/wiki_t/split_{s}.json"))["cases"]:
        for t in c["triples"]:
            ents.add(t[0]); ents.add(t[2])
low = {e.lower(): e for e in ents}
fs = fsspec.filesystem("http")
os.makedirs(f"{ROOT}/wiki_rev", exist_ok=True)
LOC = f"{ROOT}/wiki_rev/locations.json"


def retry(f, *a):
    for i in range(12):
        try:
            return f(*a)
        except Exception as e:
            time.sleep(3 + 3 * i)
    raise RuntimeError(str(a))


def open_pf(url):
    return pq.ParquetFile(fs.open(url, block_size=1 << 18))


def scan_shard(d, i):
    url = DUMPS[d].format(i)
    pf = retry(open_pf, url)
    hits = []
    for g in range(pf.metadata.num_row_groups):
        ti = retry(lambda: pf.read_row_group(g, columns=["title"])["title"].to_pylist())
        for r, t in enumerate(ti):
            e = t if t in ents else low.get(t.lower())
            if e is not None:
                hits.append((e, t, i, g, r))
    print(d, i, len(hits), flush=True)
    return d, hits


if os.path.exists(LOC):
    loc = json.load(open(LOC))
else:
    loc = {d: [] for d in DUMPS}
    with cf.ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 24) as ex:
        for d, hits in ex.map(lambda x: scan_shard(*x), [(d, i) for d in DUMPS for i in range(41)]):
            loc[d] += hits
    json.dump(loc, open(LOC, "w"))
# prefer exact-title matches; keep one location per entity and dump
best = {}
for d, hits in loc.items():
    for e, t, i, g, r in hits:
        if (d, e) not in best or t == e:
            best[(d, e)] = (t, i, g, r)
both = [e for e in ents if all((d, e) in best for d in DUMPS)]
groups = {(d, best[(d, e)][1], best[(d, e)][2]) for d in DUMPS for e in both}
print("entities", len(ents), "in both dumps", len(both), "row groups to read", len(groups), flush=True)


def paras(text):
    out = []
    for p in text.split("\n"):
        p = re.sub(r"\s+", " ", p).strip()
        if len(p) >= 80:
            out.append(p)
        if len(out) >= 10:
            break
    return out


def read_group(key):
    d, i, g = key
    pf = retry(open_pf, DUMPS[d].format(i))
    tab = retry(lambda: pf.read_row_group(g, columns=["id", "title", "text"]))
    return key, tab["id"].to_pylist(), tab["text"].to_pylist()


texts = {}
with cf.ThreadPoolExecutor(int(sys.argv[2]) if len(sys.argv) > 2 else 16) as ex:
    for n, (key, ids, tx) in enumerate(ex.map(read_group, sorted(groups))):
        texts[key] = (ids, tx)
        if n % 50 == 0:
            print("groups", n, len(groups), flush=True)
out = {}
for e in both:
    snaps = {}
    for d in DUMPS:
        t, i, g, r = best[(d, e)]
        ids, tx = texts[(d, i, g)]
        snaps[d] = {"revid": ids[r], "ts": d, "paras": paras(tx[r]), "title": t}
    if snaps["2022-03-01"]["paras"]:
        out[e] = {"title": snaps["2022-03-01"]["title"], "snaps": snaps}
json.dump(out, open(f"{ROOT}/wiki_rev/revs.json", "w"))
print("written", len(out), flush=True)
