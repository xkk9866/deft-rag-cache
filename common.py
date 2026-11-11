"""Shared utilities: paths, answer normalisation, dense retrieval index, LLM wrapper."""
import json, os, re, string, gzip, pickle, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch

ROOT = os.environ.get("EVROOT", "/data/yaokang/evcache")
DATA = f"{ROOT}/data"
RUNS = f"{ROOT}/runs"
BGE = os.environ.get("BGE", "/home/yaokang/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5/snapshots")
LLM_PATH = os.environ.get("LLM_PATH", "/home/yaokang/models/Qwen3-8B-AWQ")
QPREFIX = "Represent this sentence for searching relevant passages: "


def bge_path():
    p = BGE
    if p.endswith("snapshots"):
        p = os.path.join(p, sorted(os.listdir(p))[0])
    return p


def jdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "wt", encoding="utf-8") as f:
        json.dump(obj, f)


def jload(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def pdump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def pload(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------- answers
def normalize(s):
    s = (s or "").lower().strip()
    s = s.split("\n")[0]
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def same_answer(a, b):
    return normalize(a) == normalize(b)


NUMW = {w: str(i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve".split())}
ABSTAIN = re.compile(r"\b(unknown|not (explicitly |directly )?(mention|state|specif|provid|contain|includ|list|given)\w*|no (information|mention|data)|"
                     r"does not|do not|cannot|can't|unable to|none of the (documents|provided)|not enough information|contains an error)\b")


def canon(a):
    if not normalize(a) or ABSTAIN.search((a or "").lower()):
        return ("<abstain>",)
    return tuple(NUMW.get(t, t) for t in normalize(a).split())


import functools


@functools.lru_cache(maxsize=None)
def equiv(a, b):
    """answer equivalence used for staleness: equal normalised token sets, token containment
    (Lyon / Lyon, France), or both abstaining."""
    ca, cb = canon(a), canon(b)
    if ca == cb:
        return True
    if os.environ.get("EQUIV") == "strict":          # sensitivity analysis: only identical normalised answers
        return False
    if "<abstain>" in ca or "<abstain>" in cb:
        return False
    sa, sb = set(ca), set(cb)
    small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    return len(small) > 0 and small <= big


def em_alias(pred, golds):
    p = normalize(pred)
    return float(any(p == normalize(g) for g in golds if g))


def contains_alias(pred, golds):
    p = normalize(pred)
    return float(any(normalize(g) and (normalize(g) in p or (p and p in normalize(g))) for g in golds))


# ---------------------------------------------------------------- encoder
class Encoder:
    """bge-base encoder (CLS pooling, L2 normalised) on one GPU."""

    def __init__(self, device="cuda:0", path=None, max_len=256):
        from transformers import AutoTokenizer, AutoModel
        path = path or bge_path()
        self.tok = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.device, self.max_len = device, max_len

    @torch.no_grad()
    def encode(self, texts, bs=256, query=False):
        out = []
        for i in range(0, len(texts), bs):
            chunk = [(QPREFIX + t) if query else t for t in texts[i:i + bs]]
            b = self.tok(chunk, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h.float(), dim=-1).half())
        return torch.cat(out) if out else torch.zeros(0, 768, dtype=torch.float16, device=self.device)


class DenseIndex:
    """Exact inner-product index over a mutable corpus (insert / delete / modify).

    Deleted rows are masked; modified docs get a fresh row (old row masked) so that
    every doc version has a stable row id."""

    def __init__(self, dim=768, device="cuda:0", cap=400000):
        self.E = torch.zeros(cap, dim, dtype=torch.float16, device=device)
        self.alive = torch.zeros(cap, dtype=torch.bool, device=device)
        self.n = 0
        self.texts, self.doc_of_row = [], []      # row -> text, row -> logical doc id
        self.row_of_doc = {}                       # logical doc id -> current row
        self.device = device

    def add(self, doc_ids, texts, emb):
        k = len(texts)
        rows = list(range(self.n, self.n + k))
        self.E[self.n:self.n + k] = emb
        self.alive[self.n:self.n + k] = True
        for r, d, t in zip(rows, doc_ids, texts):
            if d in self.row_of_doc:
                self.alive[self.row_of_doc[d]] = False
            self.row_of_doc[d] = r
            self.texts.append(t)
            self.doc_of_row.append(d)
        self.n += k
        return rows

    def delete(self, doc_id):
        r = self.row_of_doc.pop(doc_id, None)
        if r is not None:
            self.alive[r] = False
        return r

    @torch.no_grad()
    def search(self, q, k=5):
        """q: [m, dim] -> (scores [m,k], rows [m,k]) with deterministic tie-break by row id."""
        s = q.float() @ self.E[:self.n].float().T
        s[:, ~self.alive[:self.n]] = -1e4
        sc, rows = torch.topk(s, k, dim=1)
        return sc, rows

    def snapshot(self):
        return {"n": self.n, "alive": self.alive[:self.n].clone(), "row_of_doc": dict(self.row_of_doc)}


# ---------------------------------------------------------------- LLM
class LLMWrap:
    def __init__(self, path=LLM_PATH, mem=0.62, max_len=int(os.environ.get("LLM_MAXLEN", 4096))):
        mem = float(os.environ.get("LLM_MEM", mem))
        from vllm import LLM, SamplingParams
        self.llm = LLM(model=path, quantization="awq_marlin", gpu_memory_utilization=mem, max_model_len=max_len, seed=0,
                       enable_prefix_caching=True,
                       max_num_seqs=int(os.environ.get("LLM_SEQS", 256)),
                       enforce_eager=bool(os.environ.get("LLM_EAGER")),
                       **({"max_num_batched_tokens": int(os.environ["LLM_BTOK"])} if os.environ.get("LLM_BTOK") else {}))
        self.tok = self.llm.get_tokenizer()
        self.SP = SamplingParams
        self.calls = 0

    def chat(self, user_msgs, max_tokens=48, system=None):
        prompts = []
        for u in user_msgs:
            msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": u}]
            prompts.append(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False))
        if not prompts:
            return []
        outs = self.llm.generate(prompts, self.SP(temperature=0, max_tokens=max_tokens), use_tqdm=False)
        self.calls += len(prompts)
        return [o.outputs[0].text.strip() for o in outs]


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, v):
        self.t[k] = self.t.get(k, 0.0) + v
