from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish


FIELDS = [
    "name",
    "phone",
    "house_flat",
    "building",
    "street",
    "landmark",
    "locality",
    "city",
    "state",
    "pincode",
]


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def extract_span_text(sp: spm.SentencePieceProcessor, ids: List[int], s: int, e: int) -> str:
    if s < 0 or e < s or s >= len(ids):
        return ""
    e = min(e, len(ids) - 1)
    text = sp.DecodeIds(ids[s : e + 1])
    text = " ".join(text.split()).strip()
    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    ap.add_argument("--raw_address", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--confidence", type=float, default=0.30)
    args = ap.parse_args()

    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(args.spm)))
    pad_id = sp.pad_id()

    sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])

    raw = normalize_hinglish(args.raw_address)
    raw = " ".join(raw.split()).strip()

    ids = sp.EncodeAsIds(raw)
    # add EOS if exists
    eos_id = sp.eos_id()
    if eos_id != -1:
        ids = ids + [eos_id]

    ids = ids[: args.max_len]
    attn = [1] * len(ids)

    if len(ids) < args.max_len:
        pad_len = args.max_len - len(ids)
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.array([attn], dtype=np.int64)

    start_logits, end_logits = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    # [1,F,T] -> [F,T]
    start_logits = start_logits[0]
    end_logits = end_logits[0]

    T_valid = int(np.sum(attn))
    ids_core = ids[:T_valid]

    parsed: Dict[str, Optional[str]] = {f: None for f in FIELDS}

    bad_name_tokens = {
        "flat","house","h","no","hno","road","rd","street","st","near","opp","behind",
        "tower","apts","apartment","society","phase","block","wing","mg","main"
    }

    def looks_like_masked_phone(s: str) -> bool:
        s2 = s.replace(" ", "")
        if "x" in s2.lower():
            return True
        digits = "".join([c for c in s2 if c.isdigit()])
        # reject real-looking phones
        if len(digits) >= 10:
            return False
        return False

    field_thresh = {f: float(args.confidence) for f in FIELDS}
    field_thresh["name"] = max(float(args.confidence), 0.55)
    field_thresh["phone"] = max(float(args.confidence), 0.65)

    for fi, field in enumerate(FIELDS):
        s = start_logits[fi, :T_valid]
        e = end_logits[fi, :T_valid]

        ps = softmax(s.astype(np.float64))
        pe = softmax(e.astype(np.float64))

        s_idx = int(np.argmax(ps))
        e_idx = int(np.argmax(pe))
        conf = float(ps[s_idx] * pe[e_idx])

        if e_idx < s_idx:
            continue
        if conf < field_thresh.get(field, float(args.confidence)):
            continue

        text = extract_span_text(sp, ids_core, s_idx, e_idx)
        if not text:
            continue

        text = " ".join(text.split()).strip()

        if field == "pincode":
            digits = "".join([c for c in text if c.isdigit()])
            text = digits if len(digits) == 6 else ""
            if not text:
                continue

        if field == "phone":
            if not looks_like_masked_phone(text):
                continue

        if field == "name":
            low = text.lower()
            toks = [t for t in low.replace(",", " ").split() if t]
            if any(t in bad_name_tokens for t in toks):
                continue
            if len(toks) == 1 and len(toks[0]) <= 3:
                continue

        parsed[field] = text

    print(json.dumps({"raw_address": raw, "parsed": parsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
