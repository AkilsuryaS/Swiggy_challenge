from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from opa.tokenization.tokenizer import SentencePieceTokenizer


_WORD = re.compile(r"[a-zA-Z0-9']+")


@dataclass
class TaskAExample:
    input_ids: List[int]
    attention_mask: List[int]
    intent_id: int
    slot_label_ids: List[int]


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_label_maps(label_maps_path: Path) -> Dict:
    return json.loads(label_maps_path.read_text(encoding="utf-8"))


def _find_value_span_words(text: str, value: str) -> Tuple[int, int] | None:
    """
    Finds the start/end word indices (inclusive) for a slot value inside text, using a
    forgiving word-token match.
    Returns (start_word_idx, end_word_idx) or None.
    """
    text_words = _WORD.findall(text.lower())
    val_words = _WORD.findall(str(value).lower())
    if not val_words:
        return None

    # exact subsequence match on word tokens
    for i in range(0, len(text_words) - len(val_words) + 1):
        if text_words[i : i + len(val_words)] == val_words:
            return i, i + len(val_words) - 1
    return None


def _sp_word_offsets(sp: SentencePieceTokenizer, text: str, max_length: int) -> Tuple[List[int], List[int], List[Tuple[int, int]]]:
    """
    Encodes text and returns:
    - input_ids (padded to max_length)
    - attention_mask
    - word_offsets: for each token position, (start_word_idx, end_word_idx) that token covers
      computed via SentencePiece's EncodeAsPieces + "alignment" by greedy word coverage.
    Practical approach:
    - Compute word list
    - Compute pieces
    - Map each piece back to a word index range based on decoded piece surface heuristics
    This is approximate but works well enough for synthetic data.
    """
    # Note: SentencePieceProcessor can give pieces; we do best-effort alignment.
    pieces = sp.sp.EncodeAsPieces(text)
    words = _WORD.findall(text.lower())

    # Build a character-to-word map using regex spans on original text (lowered)
    lower = text.lower()
    spans = [(m.start(), m.end()) for m in _WORD.finditer(lower)]
    char2word = {}
    for wi, (s, e) in enumerate(spans):
        for c in range(s, e):
            char2word[c] = wi

    # Now map each piece to word indices by locating the piece string in a running cursor.
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for p in pieces:
        surf = p.replace("▁", " ")
        surf = surf.strip()
        if not surf:
            offsets.append((-1, -1))
            continue

        # find next occurrence after cursor
        idx = lower.find(surf, cursor)
        if idx == -1:
            # fallback: don't advance; unknown mapping
            offsets.append((-1, -1))
            continue
        start = idx
        end = idx + len(surf) - 1
        cursor = idx + len(surf)

        ws = [char2word.get(c, -1) for c in range(start, end + 1)]
        ws = [w for w in ws if w != -1]
        if not ws:
            offsets.append((-1, -1))
        else:
            offsets.append((min(ws), max(ws)))

    input_ids = sp.encode(text, add_bos=False, add_eos=True, max_length=max_length)
    attn = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [sp.pad_id] * pad_len
        attn = attn + [0] * pad_len

    # Offsets correspond to pieces; but input_ids includes EOS as extra id and padding.
    # We approximate by aligning offsets to token positions excluding EOS/pad.
    # For EOS and PAD positions -> (-1,-1)
    effective_len = min(len(offsets), max_length)
    tok_offsets = offsets[:effective_len]
    # add EOS offset if it exists in sequence
    if effective_len < max_length:
        tok_offsets.append((-1, -1))
        tok_offsets = tok_offsets[:max_length]
    # pad offsets to max_length
    if len(tok_offsets) < max_length:
        tok_offsets += [(-1, -1)] * (max_length - len(tok_offsets))

    return input_ids, attn, tok_offsets


class TaskADataset(Dataset):
    def __init__(
        self,
        *,
        jsonl_path: Path,
        label_maps_path: Path,
        sp_model_path: Path,
        max_length: int = 64,
    ):
        self.rows = _read_jsonl(Path(jsonl_path))
        self.maps = _load_label_maps(Path(label_maps_path))
        self.intent2id: Dict[str, int] = self.maps["intent2id"]
        self.slot2id: Dict[str, int] = self.maps["slot2id"]
        self.slot_keys: List[str] = self.maps["slot_keys"]

        self.tokenizer = SentencePieceTokenizer(Path(sp_model_path))
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TaskAExample:
        r = self.rows[idx]
        text = r["text"]
        intent = r["intent"]
        slots: Dict[str, str] = r.get("slots", {}) or {}

        input_ids, attn, tok_word_offsets = _sp_word_offsets(self.tokenizer, text, self.max_length)

        # Build word-level spans for each slot value
        word_spans: Dict[str, Tuple[int, int]] = {}
        for k, v in slots.items():
            span = _find_value_span_words(text, v)
            if span is not None:
                word_spans[k] = span

        # Convert to token-level BIO tags
        slot_labels = ["O"] * self.max_length
        for k, (ws, we) in word_spans.items():
            for ti, (tws, twe) in enumerate(tok_word_offsets):
                if tws == -1:
                    continue
                # token overlaps word span?
                if not (twe < ws or tws > we):
                    slot_labels[ti] = f"I-{k}"

            # set B- for first token that overlaps
            for ti, (tws, twe) in enumerate(tok_word_offsets):
                if tws == -1:
                    continue
                if not (twe < ws or tws > we):
                    slot_labels[ti] = f"B-{k}"
                    break

        slot_label_ids = [self.slot2id.get(t, self.slot2id["O"]) for t in slot_labels]
        intent_id = int(self.intent2id[intent])

        return TaskAExample(
            input_ids=input_ids,
            attention_mask=attn,
            intent_id=intent_id,
            slot_label_ids=slot_label_ids,
        )


def as_tensors(ex: TaskAExample) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
        "intent_id": torch.tensor(ex.intent_id, dtype=torch.long),
        "slot_label_ids": torch.tensor(ex.slot_label_ids, dtype=torch.long),
    }
