"""Parallel ranged downloader for a Hugging Face repo via hf-mirror (per-connection throughput is throttled, so many
small ranged requests in a pool).  Every chunk is verified by length before it is written at its offset; a chunk that
fails is retried independently.  usage: pdl.py <repo> <dstdir> [workers=24] [chunk_mb=16]"""
import json, os, sys, time, urllib.request, concurrent.futures as cf

repo, dst = sys.argv[1], sys.argv[2]            # repo may be "hf_repo|ms_repo" to pull from both mirrors at once
W = int(sys.argv[3]) if len(sys.argv) > 3 else 24
CH = (int(sys.argv[4]) if len(sys.argv) > 4 else 16) * 1024 * 1024
SRC = sys.argv[5] if len(sys.argv) > 5 else "both"       # hf = hf-mirror.com, ms = modelscope.cn, both = alternate chunks
os.makedirs(dst, exist_ok=True)
KEEP = (".json", ".txt", ".safetensors", ".model", ".tiktoken")
hf_repo, ms_repo = (repo.split("|") + [repo])[:2]


def get(url, rng=None, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "pdl/1"} | ({"Range": f"bytes={rng[0]}-{rng[1]}"} if rng else {}))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


URLS = []
if SRC in ("hf", "both"):
    meta = json.loads(get(f"https://hf-mirror.com/api/models/{hf_repo}?blobs=true")[1])
    files = [(s["rfilename"], s.get("size") or 0) for s in meta["siblings"]]
    URLS.append(lambda f: f"https://hf-mirror.com/{hf_repo}/resolve/main/{f}")
if SRC in ("ms", "both"):
    meta = json.loads(get(f"https://www.modelscope.cn/api/v1/models/{ms_repo}/repo/files?Recursive=true")[1])
    files = [(x["Path"], x["Size"]) for x in meta["Data"]["Files"] if x["Type"] == "blob"]
    URLS.append(lambda f: f"https://www.modelscope.cn/models/{ms_repo}/resolve/master/{f}")
files = [(f, sz) for f, sz in files if (f.endswith((".json", ".safetensors", ".model", ".tiktoken")) or f in ("merges.txt", "vocab.json"))
         and not f.endswith("configuration.json") and "/" not in f]
print(SRC, files, flush=True)
for f, sz in files:
    p = os.path.join(dst, f)
    if os.path.exists(p) and os.path.getsize(p) == sz:
        print("have", f, flush=True); continue
    urls = [u(f) for u in URLS]
    url = urls[0]
    if sz < 4 * CH:
        for t in range(8):
            try:
                st, data = get(urls[t % len(urls)]); assert len(data) == sz or sz == 0, (len(data), sz)
                open(p, "wb").write(data); break
            except Exception as e:
                print("retry small", f, t, repr(e)[:100], flush=True); time.sleep(3)
        print("small", f, os.path.getsize(p) if os.path.exists(p) else "MISSING", flush=True); continue
    done_f = p + ".chunks"
    done = set(json.load(open(done_f))) if os.path.exists(done_f) else set()
    if not os.path.exists(p + ".part"):
        with open(p + ".part", "wb") as fo:
            fo.truncate(sz)
    fd = os.open(p + ".part", os.O_RDWR)
    chunks = [(i, i * CH, min(sz, (i + 1) * CH) - 1) for i in range((sz + CH - 1) // CH) if i not in done]
    t0, got = time.time(), 0

    def fetch(c):
        i, a, b = c
        for t in range(10):
            u = urls[(i + t) % len(urls)]                      # alternate mirrors by chunk; a failure moves to the other
            try:
                st, data = get(u, (a, b), timeout=120)
                if st in (200, 206) and len(data) == b - a + 1:
                    os.pwrite(fd, data, a); return i, len(data)
                raise ValueError(f"status {st} len {len(data)} want {b - a + 1}")
            except Exception as e:
                time.sleep(2 + 2 * t)
        return i, -1

    with cf.ThreadPoolExecutor(W) as ex:
        for i, n in ex.map(fetch, chunks):
            if n > 0:
                done.add(i); got += n
                if len(done) % 5 == 0:
                    json.dump(sorted(done), open(done_f, "w"))
                    print(f"  {f}: {len(done)}/{(sz + CH - 1) // CH} chunks, {got / (time.time() - t0) / 1e6:.2f} MB/s", flush=True)
            else:
                print("  chunk failed", i, flush=True)
    os.close(fd); json.dump(sorted(done), open(done_f, "w"))
    if len(done) == (sz + CH - 1) // CH:
        os.rename(p + ".part", p); os.remove(done_f); print("done", f, sz, flush=True)
    else:
        print("INCOMPLETE", f, len(done), flush=True); sys.exit(1)
print("PDL_DONE", repo, flush=True)
