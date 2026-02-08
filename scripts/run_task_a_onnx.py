from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish

DIGIT_RE = re.compile(r"\b(\d{1,3})\b")


def softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def load_label_maps(path: Path) -> Tuple[Dict[int, str], Dict[int, str]]:
    """
    Expected keys in label_maps.json:
      - intent_id2label / slot_id2label (new)
      - id2intent / id2slot (current)
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))

    # allow both list and dict formats
    def to_id2label(x):
        if isinstance(x, list):
            return {i: v for i, v in enumerate(x)}
        return {int(k): v for k, v in x.items()}

    intent_key = "intent_id2label" if "intent_id2label" in obj else "id2intent"
    slot_key = "slot_id2label" if "slot_id2label" in obj else "id2slot"
    intent_id2label = to_id2label(obj[intent_key])
    slot_id2label = to_id2label(obj[slot_key])
    return intent_id2label, slot_id2label


def merge_bio_spans(
    tokens: List[str],
    labels: List[str],
) -> Dict[str, str]:
    """
    Convert token-level BIO labels into {slot_name: slot_value}.
    Assumes labels like: O, B-delay_min, I-delay_min
    """
    slots: Dict[str, str] = {}

    cur_slot: Optional[str] = None
    cur_tokens: List[str] = []

    def flush():
        nonlocal cur_slot, cur_tokens
        if cur_slot and cur_tokens:
            val = " ".join(cur_tokens).strip()
            val = val.replace("▁", " ").strip()  # sentencepiece underline
            val = " ".join(val.split())
            if val:
                slots[cur_slot] = val
        cur_slot = None
        cur_tokens = []

    for tok, lab in zip(tokens, labels):
        if lab == "O" or lab is None:
            flush()
            continue

        # Allow both "B-slot" and "B_SLOT" styles
        if lab.startswith("B-") or lab.startswith("B_"):
            flush()
            cur_slot = lab[2:].replace("_", "-")
            cur_tokens = [tok]
        elif lab.startswith("I-") or lab.startswith("I_"):
            slot_name = lab[2:].replace("_", "-")
            if cur_slot == slot_name:
                cur_tokens.append(tok)
            else:
                # broken BIO sequence: start new span
                flush()
                cur_slot = slot_name
                cur_tokens = [tok]
        else:
            # unknown label format -> ignore safely
            continue

    flush()
    return slots


def fix_delay_min(slots: Dict[str, str], text: str) -> Dict[str, str]:
    """
    If intent is report_delay but delay_min slot missing, extract digits near 'min'.
    This is a safe fallback (helps demo reliability).
    """
    if "delay_min" in slots:
        # normalize numeric only
        digits = "".join([c for c in slots["delay_min"] if c.isdigit()])
        if digits:
            slots["delay_min"] = digits
        return slots

    t = text.lower()
    # Prefer numbers that are followed by "min"
    m = re.search(r"\b(\d{1,3})\s*(min|mins|minute|minutes)\b", t)
    if m:
        slots["delay_min"] = m.group(1)
        return slots

    # fallback: any number in the text (less strict)
    m2 = DIGIT_RE.search(t)
    if m2:
        slots["delay_min"] = m2.group(1)
    return slots


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    ap.add_argument("--label_maps", type=str, required=True)
    ap.add_argument("--text", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--slot_conf", type=float, default=0.0, help="set >0 to filter low-confidence slot tokens")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(args.spm)))
    pad_id = sp.pad_id()
    eos_id = sp.eos_id()

    # Load label maps
    intent_id2label, slot_id2label = load_label_maps(Path(args.label_maps))

    # Load ONNX session
    sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])

    # Normalize + tokenize
    text = normalize_hinglish(args.text)
    text = " ".join(text.split()).strip()

    ids = sp.EncodeAsIds(text)
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

    outputs = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    # We assume model outputs: intent_logits, slot_logits
    # intent_logits: [B, num_intents]
    # slot_logits:   [B, T, num_slot_labels]  (or [B, num_slot_labels, T] depending)
    if len(outputs) != 2:
        raise RuntimeError(f"Expected 2 outputs (intent_logits, slot_logits), got {len(outputs)}")

    intent_logits, slot_logits = outputs[0], outputs[1]

    # Intent
    intent_probs = softmax(intent_logits[0])
    intent_id = int(np.argmax(intent_probs))
    intent = intent_id2label.get(intent_id, str(intent_id))

    # Slot logits shape handling
    slot_logits = slot_logits[0]
    # If [num_labels, T] -> transpose to [T, num_labels]
    if slot_logits.ndim == 2 and slot_logits.shape[0] < slot_logits.shape[1]:
        # ambiguous; prefer treating as [T, C] if T==max_len
        pass
    if slot_logits.ndim == 2 and slot_logits.shape[0] == len(slot_id2label) and slot_logits.shape[1] == args.max_len:
        slot_logits = slot_logits.T  # [T, C]

    if slot_logits.ndim != 2:
        raise RuntimeError(f"slot_logits expected 2D (T,C) after reshape, got shape {slot_logits.shape}")

    T_valid = int(np.sum(attn))
    token_ids_core = ids[:T_valid]
    pieces = [sp.IdToPiece(int(t)) for t in token_ids_core]

    # Predict per token label
    pred_labels: List[str] = []
    for t in range(T_valid):
        logits_t = slot_logits[t]
        probs_t = softmax(logits_t)
        lab_id = int(np.argmax(probs_t))
        lab = slot_id2label.get(lab_id, "O")

        # Optional confidence filter (default 0.0 -> disabled)
        if args.slot_conf > 0 and float(np.max(probs_t)) < args.slot_conf:
            lab = "O"

        pred_labels.append(lab)

    slots = merge_bio_spans(pieces, pred_labels)

    # If report_delay intent but delay_min missing, add safe fallback
    if intent == "report_delay":
        slots = fix_delay_min(slots, text)

    out = {"intent": intent, "slots": slots}

    if args.debug:
        debug = {
            "text": text,
            "pieces": pieces,
            "labels": pred_labels,
            "intent_probs_top3": sorted(
                [(intent_id2label.get(i, str(i)), float(p)) for i, p in enumerate(intent_probs)],
                key=lambda x: x[1],
                reverse=True,
            )[:3],
        }
        out["_debug"] = debug

    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main()
