"""Build evolving-corpus workloads from MQuAKE (Wikidata version differences).

W1 (wiki_t)  : MQuAKE-T, real Wikidata 2021-04 -> 2023-04 changes.
W2 (wiki_cf) : MQuAKE-CF-3k-v2, counterfactual multi-edit chains (edits on any hop).

Output (per workload, per split in {train, cal, test}):
  corpus.json   : initial documents  {doc_id: text}
  pool.json     : held-out paragraphs used by benign insert events
  split_{s}.json: {"cases": [...cached questions...], "events": [...update stream...]}
Every synthetic news document is paraphrased by the LLM so that inserted text is not template-matched.
"""
import argparse, random, collections, re
import pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--wl", required=True, choices=["wiki_t", "wiki_cf"])
ap.add_argument("--n_bg", type=int, default=30000)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_cases", type=int, default=2400)
args = ap.parse_args()
rng = random.Random(args.seed)
OUT = f"{DATA}/{args.wl}"
src = {"wiki_t": "MQuAKE-T.json", "wiki_cf": "MQuAKE-CF-3k-v2.json"}[args.wl]
cases = jload(f"{ROOT}/repos/MQuAKE/datasets/{src}")

# ------------------------------------------------ consistent case subset (one new value per (s,r))
edit_val, chosen = {}, []
rng.shuffle(cases)
for c in cases:
    ok = True
    for r in c["requested_rewrite"]:
        key = (r["subject"], r["relation_id"])
        if key in edit_val and edit_val[key] != r["target_new"]["str"]:
            ok = False
    if not ok:
        continue
    for r in c["requested_rewrite"]:
        edit_val[(r["subject"], r["relation_id"])] = r["target_new"]["str"]
    chosen.append(c)
    if len(chosen) >= args.max_cases:
        break
cases = chosen
print("cases", len(cases))

# ------------------------------------------------ facts (triples) with natural cloze sentences
fact_text, fact_rid = {}, {}   # (s, rel_label, o) -> sentence / wikidata relation id
for c in cases:
    for t, tid, h in zip(c["orig"]["triples_labeled"], c["orig"]["triples"], c["single_hops"]):
        fact_text.setdefault(tuple(t), h["cloze"].strip() + " " + t[2] + ".")
        fact_rid[tuple(t)] = tid[1]
facts = sorted(fact_text)
fid = {f: f"F{i}" for i, f in enumerate(facts)}

# edits: (s, rid) -> new value, with sentence from the rewrite prompt
edits = {}
for c in cases:
    for r in c["requested_rewrite"]:
        key = (r["subject"], r["relation_id"])
        if key not in edits:
            edits[key] = {"subject": r["subject"], "rid": r["relation_id"], "new": r["target_new"]["str"],
                          "old": r["target_true"]["str"], "sent": r["prompt"].format(r["subject"]) + " " + r["target_new"]["str"] + ".",
                          "question": r["question"]}
edit_keys = sorted(edits)
print("unique edits", len(edit_keys), "facts", len(facts))

# ------------------------------------------------ split cases by edit-connected components
parent = {}
def find(x):
    while parent.setdefault(x, x) != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for c in cases:
    ks = [("E",) + (r["subject"], r["relation_id"]) for r in c["requested_rewrite"]]
    for k in ks[1:]:
        parent[find(k)] = find(ks[0])
comp_cases = collections.defaultdict(list)
for c in cases:
    k = ("E", c["requested_rewrite"][0]["subject"], c["requested_rewrite"][0]["relation_id"])
    comp_cases[find(k)].append(c)
comps = sorted(comp_cases, key=lambda k: str(k)); rng.shuffle(comps)
ratios = {"wiki_t": (0.20, 0.35, 0.45), "wiki_cf": (0.40, 0.15, 0.45)}[args.wl]
tot = len(cases); acc = 0; split_of = {}
for k in comps:
    frac = acc / tot
    s = "train" if frac < ratios[0] else ("cal" if frac < ratios[0] + ratios[1] else "test")
    for c in comp_cases[k]:
        split_of[c["case_id"]] = s
    acc += len(comp_cases[k])
print(collections.Counter(split_of.values()))

# ------------------------------------------------ background Wikipedia paragraphs (HotpotQA, 2017 dump)
df = pd.read_parquet(f"{DATA}/hotpot/distractor/train-00000-of-00002.parquet")
paras = {}
for ctx in df["context"]:
    for t, ss in zip(ctx["title"], ctx["sentences"]):
        if t not in paras:
            paras[t] = t + ": " + "".join(ss).strip()
titles = sorted(paras); rng.shuffle(titles)
ents = set()
for f in facts:
    ents.add(f[0]); ents.add(f[2])
matched = [t for t in titles if t in ents]
rng.shuffle(matched)
bg_titles = set(matched[: len(matched) // 2]) | set([t for t in titles if t not in ents][: args.n_bg])
held_matched = [t for t in matched[len(matched) // 2:]]
held_rand = [t for t in titles if t not in ents and t not in bg_titles][:20000]
print("bg", len(bg_titles), "held matched", len(held_matched), "held rand", len(held_rand))

corpus = {fid[f]: "(2021) " + fact_text[f] for f in facts}
for t in bg_titles:
    corpus["B:" + t] = "(2017) " + paras[t]

# ------------------------------------------------ LLM paraphrase of synthetic news
llm = LLMWrap(mem=0.55)
PARA = ("Rewrite the following fact as one short news-style sentence published in {year}. Keep every name exactly and keep "
        "the meaning; {extra}do not add any other facts.\nFact: {fact}\nSentence:")
def paraphrase(items):
    msgs = [PARA.format(year=y, fact=f, extra=e) for f, y, e in items]
    outs = llm.chat(msgs, max_tokens=80)
    res = []
    for (f, y, e), o in zip(items, outs):
        o = o.split("\n")[0].strip().strip('"')
        res.append(o if len(o) > 10 else f)
    return res

def case_rec(c):
    return {"cid": c["case_id"], "q": c["questions"][0], "paraphrases": c["questions"][1:],
            "answer": c["answer"], "answer_alias": [c["answer"]] + c.get("answer_alias", []),
            "new_answer": c["new_answer"], "new_answer_alias": [c["new_answer"]] + c.get("new_answer_alias", []),
            "hops": [h["question"] for h in c["single_hops"]], "triples": c["orig"]["triples_labeled"],
            "edits": [(r["subject"], r["relation_id"]) for r in c["requested_rewrite"]]}

for split in ["train", "cal", "test"]:
    sc = [c for c in cases if split_of[c["case_id"]] == split]
    if not sc:
        continue
    s_edits = sorted({(r["subject"], r["relation_id"]) for c in sc for r in c["requested_rewrite"]})
    s_facts = sorted({tuple(t) for c in sc for t in c["orig"]["triples_labeled"]})
    s_ents = {f[0] for f in s_facts} | {f[2] for f in s_facts}
    nI = len(s_edits)
    ev = []
    # true updates (+ a later duplicate report of the same update)
    items = [(edits[k]["sent"], "2023", "") for k in s_edits]
    news = paraphrase(items)
    dup = paraphrase([(edits[k]["sent"], "2023", "phrase it differently, ") for k in s_edits])
    for k, n1, n2 in zip(s_edits, news, dup):
        ev.append({"op": "insert", "kind": "update", "doc": f"U:{k[0]}|{k[1]}", "text": "(2023) " + n1, "edit": list(k)})
        ev.append({"op": "insert", "kind": "update_dup", "doc": f"U2:{k[0]}|{k[1]}", "text": "(2023) " + n2, "edit": list(k)})
    # restatements of unchanged facts (same slot, same value)
    stable = [f for f in s_facts if (f[0], fact_rid[f]) not in edits]   # facts that never change
    rs = list(stable)
    rng.shuffle(rs)
    rs = rs[: 2 * nI]
    outs = paraphrase([(fact_text[f], "2023", "say that it is still the case, ") for f in rs])
    for f, o in zip(rs, outs):
        ev.append({"op": "insert", "kind": "restate", "doc": "R:" + fid[f], "text": "(2023) " + o})
    # historical values (same slot, different but past value) -- MQuAKE-T provides answer_extended
    hist = []
    for c in sc:
        last = c["orig"]["triples_labeled"][-1]
        for old in c.get("answer_extended", [])[1:3]:
            hist.append((last[0], last[1], old, c["single_hops"][-1]["cloze"]))
    hist = sorted(set(hist)); rng.shuffle(hist); hist = hist[: int(1.5 * nI)]
    if hist:
        outs = paraphrase([(f"Before {2021}, {h[2]} was formerly (no longer) the {h[1]} of {h[0]}.", "2023", "make clear it is a past role, ") for h in hist])
        for h, o in zip(hist, outs):
            ev.append({"op": "insert", "kind": "historical", "doc": f"H:{h[0]}|{h[1]}|{h[2]}", "text": "(2023) " + o})
    # same-entity paragraphs and random paragraphs (held out from the corpus)
    sib = [t for t in held_matched if t in s_ents][: 2 * nI]
    for t in sib:
        ev.append({"op": "insert", "kind": "sibling", "doc": "B:" + t, "text": "(2023) " + paras[t]})
    rnd = rng.sample(held_rand, min(len(held_rand), 3 * nI))
    for t in rnd:
        ev.append({"op": "insert", "kind": "random", "doc": "B:" + t, "text": "(2023) " + paras[t]})
    # in-place modifications: benign rewording or value correction of an existing fact doc
    mf = rng.sample(stable, min(len(stable), nI))
    objs_by_rel = collections.defaultdict(list)
    for f in facts:
        objs_by_rel[f[1]].append(f[2])
    benign = mf[: len(mf) // 2]; valchg = mf[len(mf) // 2:]
    outs = paraphrase([(fact_text[f], "2021", "") for f in benign])
    for f, o in zip(benign, outs):
        ev.append({"op": "modify", "kind": "modify_benign", "doc": fid[f], "text": "(2021) " + o})
    for f in valchg:
        alt = [o for o in objs_by_rel[f[1]] if o != f[2]]
        if not alt:
            continue
        o2 = rng.choice(alt)
        ev.append({"op": "modify", "kind": "modify_value", "doc": fid[f], "text": "(2021) " + fact_text[f][: -len(f[2]) - 1] + o2 + "."})
    # deletions: cited facts and background paragraphs
    for f in rng.sample(s_facts, min(len(s_facts), nI // 2)):
        ev.append({"op": "delete", "kind": "delete_fact", "doc": fid[f]})
    for t in rng.sample(sorted(bg_titles), nI // 2):
        ev.append({"op": "delete", "kind": "delete_bg", "doc": "B:" + t})
    # order: duplicates of an update always come after the update
    rng.shuffle(ev)
    pos = {e["doc"]: i for i, e in enumerate(ev)}
    for e in ev:
        if e["kind"] == "update_dup":
            u = "U:" + e["doc"][3:]
            if pos[u] > pos[e["doc"]]:
                i, j = pos[u], pos[e["doc"]]
                ev[i], ev[j] = ev[j], ev[i]
                pos[ev[i]["doc"]], pos[ev[j]["doc"]] = i, j
    # a doc may be touched by several events; keep the first only
    seen, ev2 = set(), []
    for e in ev:
        if e["doc"] in seen:
            continue
        seen.add(e["doc"]); ev2.append(e)
    rec = {"cases": [case_rec(c) for c in sc], "events": ev2}
    jdump(rec, f"{OUT}/split_{split}.json")
    print(split, "cases", len(sc), "events", len(ev2), collections.Counter(e["kind"] for e in ev2))

jdump(corpus, f"{OUT}/corpus.json")
print("corpus", len(corpus))
