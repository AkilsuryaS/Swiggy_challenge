from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish


def load_label_maps(label_maps_path: Path) -> Dict:
    maps = json.loads(label_maps_path.read_text(encoding="utf-8"))

    # json serialization often turns int keys into strings
    if isinstance(next(iter(maps["id2intent"].keys())), str):
        maps["id2intent"] = {int(k): v for k, v in maps["id2intent"].items()}
    if isinstance(next(iter(maps["id2slot"].keys())), str):
        maps["id2slot"] = {int(k): v for k, v in maps["id2slot"].items()}

    return maps


def encode(sp: spm.SentencePieceProcessor, text: str, max_len: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      input_ids: [1, T] int64
      attention_mask: [1, T] int64
      pieces: list of sentencepiece pieces (for slot span reconstruction)
    """
    pieces = sp.EncodeAsPieces(text)
    ids = sp.EncodeAsIds(text)

    # add EOS (SentencePiece has eos_id)
    eos_id = sp.eos_id()
    ids = ids + ([eos_id] if eos_id != -1 else [])

    ids = ids[:max_len]
    attn = [1] * len(ids)

    pad_id = sp.pad_id()
    if len(ids) < max_len:
        pad_len = max_len - len(ids)
        ids = ids + [pad_id] * pad_len
        attn = attn + [0] * pad_len

    input_ids = np.array([ids], dtype=np.int64)
    attention_mask = np.array([attn], dtype=np.int64)

    return input_ids, attention_mask, pieces


def decode_slots(sp: spm.SentencePieceProcessor, text: str, pieces: List[str], slot_ids: List[int], id2slot: Dict[int, str], attn: List[int]) -> Dict[str, str]:
    """
    Convert predicted BIO tags into slots dict.
    We rebuild slot values by joining SentencePiece surface forms inside BIO spans.
    """
    def clean_piece(p: str) -> str:
        return p.replace("▁", " ").strip()

    n = min(len(pieces), len(slot_ids))
    spans: Dict[str, List[str]] = {}
    active_key: Optional[str] = None

    for i in range(n):
        if attn[i] == 0:
            break
        tag = id2slot[int(slot_ids[i])]
        piece = clean_piece(pieces[i])

        if tag == "O":
            active_key = None
            continue

        if tag.startswith("B-"):
            active_key = tag[2:]
            spans.setdefault(active_key, [])
            if piece:
                spans[active_key].append(piece)
        elif tag.startswith("I-"):
            key = tag[2:]
            if active_key != key:
                active_key = key
                spans.setdefault(active_key, [])
            if piece:
                spans[active_key].append(piece)

    out: Dict[str, str] = {}
    for k, parts in spans.items():
        val = " ".join(parts).strip()
        val = " ".join(val.split())
        if val:
            out[k] = val

    # normalize known slots
    if "delay_min" in out:
        digits = "".join([c for c in out["delay_min"] if c.isdigit()])
        if digits:
            out["delay_min"] = digits
    if "order" in out:
        v = out["order"].lower()
        if "next" in v:
            out["order"] = "next"
        elif "prev" in v or "previous" in v:
            out["order"] = "previous"
        elif "current" in v or "this" in v:
            out["order"] = "current"
    if "issue" in out:
        v = out["issue"].lower()
        if "missing" in v:
            out["issue"] = "item_missing"
        elif "delay" in v or "late" in v:
            out["issue"] = "restaurant_delay"
        elif "address" in v or "location" in v:
            out["issue"] = "wrong_address"
        elif "pay" in v or "cash" in v or "upi" in v:
            out["issue"] = "payment"
        else:
            out["issue"] = "other"

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True, help="Quantized ONNX path (recommended: *.int8.onnx)")
    ap.add_argument("--spm", type=str, required=True, help="SentencePiece model path")
    ap.add_argument("--label_maps", type=str, required=True, help="label_maps.json path")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    spm_path = Path(args.spm)
    label_maps_path = Path(args.label_maps)

    maps = load_label_maps(label_maps_path)
    id2intent = maps["id2intent"]
    id2slot = maps["id2slot"]

    sp = spm.SentencePieceProcessor()
    sp.Load(str(spm_path))

    # normalize text like training
    text = normalize_hinglish(args.text)

    input_ids, attention_mask, pieces = encode(sp, text, args.max_len)

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    intent_logits, slot_logits = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    intent_id = int(np.argmax(intent_logits, axis=-1)[0])
    intent = id2intent[intent_id]

    slot_ids = np.argmax(slot_logits, axis=-1)[0].tolist()  # [T]
    attn = attention_mask[0].tolist()

    slots = decode_slots(sp, text, pieces, slot_ids, id2slot, attn)

    print(json.dumps({"intent": intent, "slots": slots}, ensure_ascii=False))


if __name__ == "__main__":
    main()
