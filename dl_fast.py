"""multi-connection download of the two Wikipedia dumps via hf_transfer (HF_ENDPOINT=hf-mirror)."""
import os, sys, time, shutil
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import concurrent.futures as cf
from huggingface_hub import HfApi, hf_hub_download

D = "/data/yaokang/evcache/data/wikidumps"
api = HfApi()


def one(repo, f, out):
    dst = f"{out}/{os.path.basename(f)}"
    if os.path.exists(dst):
        return
    for i in range(20):
        try:
            p = hf_hub_download(repo, f, repo_type="dataset", local_dir=f"{D}/raw")
            shutil.move(p, dst); return
        except Exception as e:
            print("retry", f, i, str(e)[:120], flush=True); time.sleep(10)


for repo, pre, out in [("wikimedia/wikipedia", "20231101.en/", f"{D}/20231101"),
                       ("legacy-datasets/wikipedia", "data/20220301.en/", f"{D}/20220301")]:
    os.makedirs(out, exist_ok=True)
    fs = [f for f in api.list_repo_files(repo, repo_type="dataset") if f.startswith(pre) and f.endswith(".parquet")]
    t = time.time()
    with cf.ThreadPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 4) as ex:
        list(ex.map(lambda f: one(repo, f, out), fs))
    print(repo, len(fs), "done", f"{time.time() - t:.0f}s", flush=True)
