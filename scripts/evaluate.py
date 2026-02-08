from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_label_maps(path: Path) -> Dict:
    maps = json.loads(path.read_text(encoding="utf-8"))
    # JSON turns int keys into strings
    if isinstance(next(iter(maps["id2intent"].keys())), str):
        maps["id2intent"] = {int(k): v for k, v in maps["id2intent"].items()}
    if isinstance(next(iter(maps["id2slot"].keys())), str):
        maps["id2slot"] = {int(k): v for k, v in maps["id2slot"].items()}
    return maps


def encode(
    sp: spm.SentencePieceProcessor,
    text: str,
    max_len: int,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    pieces = sp.EncodeAsPieces(text)
    ids = sp.EncodeAsIds(text)

    eos_id = sp.eos_id()
    if eos_id != -1:
        ids = ids + [eos_id]

    ids = ids[:max_len]
    attn = [1] * len(ids)

    pad_id = sp.pad_id()
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len

    return (
        np.array([ids], dtype=np.int64),
        np.array([attn], dtype=np.int64),
        pieces,
    )


def decode_slots(
    pieces: List[str],
    slot_ids: List[int],
    attn: List[int],
    id2slot: Dict[int, str],
) -> Dict[str, str]:
    def clean(p: str) -> str:
        return p.replace("▁", " ").strip()

    spans: Dict[str, List[str]] = {}
    active = None

    n = min(len(pieces), len(slot_ids))
    for i in range(n):
        if attn[i] == 0:
            break
        tag = id2slot[int(slot_ids[i])]
        piece = clean(pieces[i])

        if tag == "O":
            active = None
            continue

        if tag.startswith("B-"):
            active = tag[2:]
            spans.setdefault(active, [])
            if piece:
                spans[active].append(piece)
        elif tag.startswith("I-"):
            key = tag[2:]
            if active != key:
                active = key
                spans.setdefault(active, [])
            if piece:
                spans[active].append(piece)

    out: Dict[str, str] = {}
    for k, parts in spans.items():
        val = " ".join(parts).strip()
        val = " ".join(val.split())
        if val:
            out[k] = val

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    ap.add_argument("--label_maps", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="outputs/qualitative_examples/task_a.md")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    onnx_path = Path(args.onnx)
    test_path = Path(args.test_jsonl)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(test_path)
    maps = load_label_maps(Path(args.label_maps))
    id2intent = maps["id2intent"]
    id2slot = maps["id2slot"]

    sp = spm.SentencePieceProcessor()
    sp.Load(args.spm)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    samples = random.sample(rows, min(args.num_examples, len(rows)))

    lines: List[str] = []
    lines.append("# Task A – Qualitative Examples\n")
    lines.append(
        f"_Model_: INT8 ONNX | _Examples_: {len(samples)} | _Max length_: {args.max_len}\n"
    )

    for i, row in enumerate(samples, start=1):
        text = normalize_hinglish(row["text"])
        gt_intent = row["intent"]
        gt_slots = row.get("slots", {})

        input_ids, attention_mask, pieces = encode(sp, text, args.max_len)
        intent_logits, slot_logits = sess.run(
            None,
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        pred_intent_id = int(np.argmax(intent_logits, axis=-1)[0])
        pred_intent = id2intent[pred_intent_id]

        slot_ids = np.argmax(slot_logits, axis=-1)[0].tolist()
        attn = attention_mask[0].tolist()
        pred_slots = decode_slots(pieces, slot_ids, attn, id2slot)

        lines.append(f"## Example {i}\n")
        lines.append(f"**Input**: `{row['text']}`\n")
        lines.append("**Ground Truth**:\n")
        lines.append(f"- Intent: `{gt_intent}`\n")
        lines.append(f"- Slots: `{json.dumps(gt_slots, ensure_ascii=False)}`\n")
        lines.append("**Model Output**:\n")
        lines.append(f"- Intent: `{pred_intent}`\n")
        lines.append(f"- Slots: `{json.dumps(pred_slots, ensure_ascii=False)}`\n")

        if gt_intent == pred_intent:
            lines.append("✅ Intent correct\n")
        else:
            lines.append("❌ Intent incorrect\n")

        lines.append("\n---\n")

    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ Wrote qualitative examples to: {out_md}")


if __name__ == "__main__":
    main()
