"""Deterministic multi-hop RAG pipeline (self-ask / IRCoT style) that records its full retrieval trace.

trace = [{"subq", "qemb"(row in key store), "rows", "docs", "scores", "tau", "ans"}...]
The final answer depends on the corpus only through the per-hop top-k sets (Trace Invariance, Thm. 1).
"""
import re
import torch
from common import *

DECOMP = """Break a multi-hop question into simple single-fact sub-questions, asked one at a time. Every sub-question must name concrete entities (reuse earlier answers, never pronouns).
Given the question and the sub-questions answered so far, output exactly one line: either the next sub-question as "SUB: <sub-question>", or, once the answers so far determine the final answer, "FINAL: <short answer>".

Example 1
Question: Who is the head of state of the country where Rainn Wilson holds citizenship?
Answered so far:
(1) What is the country of citizenship of Rainn Wilson? -> United States of America
Next: SUB: Who is the head of state of United States of America?

Example 2
Question: What is the capital of the country where the sport played by Ronaldinho originated?
Answered so far:
(1) Which sport is Ronaldinho associated with? -> association football
(2) Which country was association football created in? -> England
(3) What is the capital of England? -> London
Next: FINAL: London

Example 3
Question: Which magazine was started first, Arthur's Magazine or First for Women?
Answered so far:
(1) When was Arthur's Magazine started? -> 1844
Next: SUB: When was First for Women started?

Example 4
Question: The director of the film Wild Strawberries was born in which city?
Answered so far:
(nothing yet)
Next: SUB: Who directed the film Wild Strawberries?

Now your turn.
Question: {q}
Answered so far:
{hist}
Next:"""

HOP = """Answer the question using the documents below. Each document starts with its year in parentheses; when documents disagree, trust the most recent one. Reply with only the answer (a short entity name, date or phrase), no explanation.

Documents:
{docs}

Question: {q}
Answer:"""


def parse_decomp(o):
    o = o.strip().split("\n")[0]
    o = re.sub(r"^Next:\s*", "", o)
    if o.upper().startswith("FINAL:"):
        return "final", o[6:].strip()
    if o.upper().startswith("SUB:"):
        return "sub", o[4:].strip()
    return "sub", o.strip()


def clean_ans(a):
    a = a.strip().split("\n")[0].strip().strip('"').rstrip(".")
    a = re.sub(r"^(Answer|A):\s*", "", a)
    return a[:120]


class RAG:
    def __init__(self, llm, enc, index, k=5, maxh=4):
        self.llm, self.enc, self.index, self.k, self.maxh = llm, enc, index, k, maxh
        self.qcache = {}      # sub-question text -> embedding (deterministic)

    def embed_q(self, qs):
        new = [q for q in dict.fromkeys(qs) if q not in self.qcache]
        if new:
            E = self.enc.encode(new, query=True)
            for q, e in zip(new, E):
                self.qcache[q] = e
        return torch.stack([self.qcache[q] for q in qs]) if qs else None

    def hop_answer_prompts(self, subqs, rows_list):
        ps = []
        for q, rows in zip(subqs, rows_list):
            docs = "\n".join(f"[{i+1}] {self.index.texts[r]}" for i, r in enumerate(rows))
            ps.append(HOP.format(docs=docs, q=q))
        return ps

    def run(self, questions):
        n = len(questions)
        traces = [[] for _ in range(n)]
        final = [None] * n
        active = list(range(n))
        for hop in range(self.maxh + 1):
            if not active:
                break
            msgs = []
            for i in active:
                hist = "\n".join(f"({j+1}) {h['subq']} -> {h['ans']}" for j, h in enumerate(traces[i])) or "(nothing yet)"
                msgs.append(DECOMP.format(q=questions[i], hist=hist))
            outs = self.llm.chat(msgs, max_tokens=48)
            subs, nxt = [], []
            for i, o in zip(active, outs):
                kind, txt = parse_decomp(o)
                repeated = any(txt.lower() == h["subq"].lower() for h in traces[i])
                if kind == "final" or hop == self.maxh or repeated or not txt:
                    final[i] = clean_ans(txt) if kind == "final" and txt else (traces[i][-1]["ans"] if traces[i] else clean_ans(txt))
                else:
                    subs.append((i, txt))
            if not subs:
                break
            qs = [t for _, t in subs]
            Q = self.embed_q(qs)
            sc, rows = self.index.search(Q, self.k)
            sc, rows = sc.cpu().tolist(), rows.cpu().tolist()
            ans = self.llm.chat(self.hop_answer_prompts(qs, rows), max_tokens=32)
            for (i, q), s, r, a in zip(subs, sc, rows, ans):
                traces[i].append({"subq": q, "rows": r, "docs": [self.index.doc_of_row[x] for x in r], "scores": s,
                                  "tau": s[-1], "ans": clean_ans(a)})
                nxt.append(i)
            active = nxt
        for i in range(n):
            if final[i] is None:
                final[i] = traces[i][-1]["ans"] if traces[i] else ""
        return [{"answer": final[i], "trace": traces[i]} for i in range(n)]

    def llm_calls_of(self, res):
        """LLM invocations needed to (re)compute one entry: one decomposition per hop (+1) and one answer per hop."""
        h = len(res["trace"])
        return 2 * h + 1
