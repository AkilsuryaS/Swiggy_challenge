from __future__ import annotations

import argparse
import json
import random
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


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def extract_span_text(sp: spm.SentencePieceProcessor, ids: List[int], s: int, e: int) -> str:
    if s < 0 or e < s or s >= len(ids):
        return ""
    e = min(e, len(ids) - 1)
    text = sp.DecodeIds(ids[s : e + 1])
    return " ".join(text.split()).strip()


def looks_like_masked_phone(s: str) -> bool:
    s2 = s.replace(" ", "")
    if "x" in s2.lower():
        return True
    digits = "".join([c for c in s2 if c.isdigit()])
    if len(digits) >= 10:
        return False
    return False


def predict_one(
    *,
    sess: ort.InferenceSession,
    sp: spm.SentencePieceProcessor,
    raw_address: str,
    max_len: int,
    confidence: float,
) -> Dict[str, Optional[str]]:
    bad_name_tokens = {
        "flat","house","h","no","hno","road","rd","street","st","near","opp","behind",
        "tower","apts","apartment","society","phase","block","wing","mg","main"
    }
    field_thresh = {f: float(confidence) for f in FIELDS}
    field_thresh["name"] = max(float(confidence), 0.55)
    field_thresh["phone"] = max(float(confidence), 0.65)

    pad_id = sp.pad_id()
    eos_id = sp.eos_id()

    raw = normalize_hinglish(raw_address)
    raw = " ".join(raw.split()).strip()

    ids = sp.EncodeAsIds(raw)
    if eos_id != -1:
        ids = ids + [eos_id]

    ids = ids[:max_len]
    attn = [1] * len(ids)

    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.array([attn], dtype=np.int64)

    start_logits, end_logits = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
    start_logits = start_logits[0]
    end_logits = end_logits[0]

    T_valid = int(np.sum(attn))
    ids_core = ids[:T_valid]

    parsed: Dict[str, Optional[str]] = {f: None for f in FIELDS}

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
        if conf < field_thresh.get(field, float(confidence)):
            continue

        text = extract_span_text(sp, ids_core, s_idx, e_idx)
        if not text:
            continue

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

    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="outputs/qualitative_examples/task_c.md")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--confidence", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    rows = read_jsonl(Path(args.test_jsonl))
    samples = random.sample(rows, min(args.num_examples, len(rows)))

    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(args.spm)))

    sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    md: List[str] = []
    md.append("# Task C – Qualitative Examples\n")
    md.append(f"_Model_: ONNX | _Examples_: {len(samples)} | _Max len_: {args.max_len}\n")

    for i, row in enumerate(samples, start=1):
        raw = row["raw_address"]
        gt = row["parsed"]
        pred = predict_one(
            sess=sess,
            sp=sp,
            raw_address=raw,
            max_len=args.max_len,
            confidence=float(args.confidence),
        )

        md.append(f"## Example {i}\n")
        md.append(f"**Raw**: `{raw}`\n")
        md.append("**Ground Truth Parsed**:\n")
        md.append("```json")
        md.append(json.dumps(gt, ensure_ascii=False, indent=2))
        md.append("```\n")
        md.append("**Model Parsed**:\n")
        md.append("```json")
        md.append(json.dumps(pred, ensure_ascii=False, indent=2))
        md.append("```\n")
        md.append("---\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Wrote Task C qualitative examples to: {out_md}")


if __name__ == "__main__":
    main()
