"""Published, open-source grounding verifiers used as verify-on-hit baselines (loaded from their released checkpoints):
  MiniCheck (Tang, Laban, Durrett, EMNLP 2024), checkpoint lytang/MiniCheck-Flan-T5-Large: input
      "premise: <chunk> hypothesis: <claim>", one decoder step, P(label 1) over the {0, 1} label tokens; the evidence is
      split into chunks and the support score is the maximum over chunks (the released inference procedure).
  HHEM-2.1-open (Vectara hallucination evaluation model, flan-t5-base): consistency score of (premise, hypothesis).
Both return P(the claim is supported by the evidence)."""
import os, torch
from common import ROOT

CLAIM = "The answer to the question '{q}' is {a}."


class MiniCheck:
    def __init__(self, path=f"{ROOT}/models/MiniCheck-Flan-T5-Large", chunk_words=400):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = AutoModelForSeq2SeqLM.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
        # first decoding step: label "1" is the single token "▁1"; label "0" starts with "▁" (it is split into "▁", "0")
        self.ids = [self.tok("0", add_special_tokens=False).input_ids[0], self.tok("1", add_special_tokens=False).input_ids[0]]
        self.cw = chunk_words

    def chunks(self, docs):
        out, cur = [], []
        for d in docs:
            w = d.split()
            if cur and len(cur) + len(w) > self.cw:
                out.append(" ".join(cur)); cur = []
            cur += w[:self.cw]
        if cur:
            out.append(" ".join(cur))
        return out

    @torch.no_grad()
    def support(self, pairs, bs=16):
        """pairs: list of (docs, claim) -> list of P(supported), max over evidence chunks"""
        flat, own = [], []
        for i, (docs, claim) in enumerate(pairs):
            for ch in self.chunks(docs) or [""]:
                flat.append(f"premise: {ch} hypothesis: {claim}"); own.append(i)
        p = []
        for i in range(0, len(flat), bs):
            b = self.tok(flat[i:i + bs], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            dec = torch.full((b.input_ids.shape[0], 1), self.m.config.decoder_start_token_id, device="cuda")
            lg = self.m(**b, decoder_input_ids=dec).logits[:, 0, self.ids].float()
            p += torch.softmax(lg, -1)[:, 1].tolist()
        out = [0.0] * len(pairs)
        for i, x in zip(own, p):
            out[i] = max(out[i], x)
        return out


class HHEM:
    def __init__(self, path=f"{ROOT}/models/hallucination_evaluation_model", base=f"{ROOT}/models/flan-t5-base"):
        from transformers import AutoConfig, AutoModelForSequenceClassification
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
        cfg.foundation = base                              # local copy of the flan-t5-base foundation (no hub access)
        self.m = AutoModelForSequenceClassification.from_pretrained(path, config=cfg, trust_remote_code=True).cuda().eval()

    @torch.no_grad()
    def support(self, pairs, bs=8):
        out = []
        for i in range(0, len(pairs), bs):
            out += [float(x) for x in self.m.predict([("\n".join(d), c) for d, c in pairs[i:i + bs]])]
        return out
