from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from opa.tokenization.tokenizer import SentencePieceTokenizer
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


@dataclass
class TaskCExample:
    input_ids: List[int]
    attention_mask: List[int]
    # span targets per field
    start_targets: List[int]   # [F]
    end_targets: List[int]     # [F]
    field_mask: List[int]      # [F] 1 if present and aligned, else 0
    # MLM
    mlm_input_ids: List[int]
    mlm_labels: List[int]      # [T] token id or pad_id(ignore)


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _find_subsequence(haystack: List[int], needle: List[int]) -> Optional[int]:
    """Return start index where needle occurs in haystack, else None."""
    if not needle or len(needle) > len(haystack):
        return None
    # naive search is fine for short sequences
    for i in range(0, len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def _make_mlm_inputs(
    ids: List[int],
    *,
    pad_id: int,
    mask_id: Optional[int],
    vocab_size: int,
    rng: random.Random,
    p_mask: float = 0.15,
) -> Tuple[List[int], List[int]]:
    """
    Standard MLM corruption:
      - 15% tokens considered
      - of those: 80% [MASK], 10% random, 10% unchanged
    If SentencePiece doesn't have a mask token, we still do random/unchanged.
    """
    x = ids.copy()
    labels = [pad_id] * len(ids)

    for i, tid in enumerate(ids):
        if tid == pad_id:
            continue
        if rng.random() > p_mask:
            continue

        labels[i] = tid

        r = rng.random()
        if mask_id is not None and mask_id >= 0 and r < 0.80:
            x[i] = mask_id
        elif r < 0.90:
            x[i] = rng.randrange(0, vocab_size)
        else:
            x[i] = tid

    return x, labels


class TaskCDataset(Dataset):
    """
    Reads:
      {"raw_address": "...", "parsed": {...fields...}}

    Builds token-level span targets by:
      - encoding raw_address to ids
      - encoding each field value to ids
      - finding field-id subsequence within raw ids
    """

    def __init__(
        self,
        *,
        jsonl_path: Path,
        sp_model_path: Path,
        max_length: int = 128,
        seed: int = 1337,
        mlm_prob: float = 0.15,
    ):
        self.rows = _read_jsonl(Path(jsonl_path))
        self.tok = SentencePieceTokenizer(Path(sp_model_path))
        self.max_length = int(max_length)
        self.rng = random.Random(int(seed))
        self.mlm_prob = float(mlm_prob)

        # Try to find a usable mask token id; if not present, set None
        # Common: "<mask>" or "[MASK]" rarely exists in SentencePiece.
        self.mask_id = None
        for cand in ["<mask>", "[MASK]", "<MASK>"]:
            try:
                cid = self.tok.sp.PieceToId(cand)
                if cid != -1:
                    self.mask_id = cid
                    break
            except Exception:
                pass

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TaskCExample:
        r = self.rows[idx]
        raw = normalize_hinglish(r["raw_address"])
        raw = " ".join(raw.split()).strip()

        parsed: Dict[str, Optional[str]] = r["parsed"]

        # Encode raw address (keep EOS for stability)
        raw_ids = self.tok.encode(raw, add_bos=False, add_eos=True, max_length=self.max_length)
        attn = [1] * len(raw_ids)

        # Pad
        if len(raw_ids) < self.max_length:
            pad_len = self.max_length - len(raw_ids)
            raw_ids = raw_ids + [self.tok.pad_id] * pad_len
            attn = attn + [0] * pad_len

        # Prepare span targets
        start_t = [0] * len(FIELDS)
        end_t = [0] * len(FIELDS)
        fmask = [0] * len(FIELDS)

        # Search only in non-pad portion (exclude padding)
        raw_core = [tid for tid, m in zip(raw_ids, attn) if m == 1]

        for fi, field in enumerate(FIELDS):
            val = parsed.get(field, None)
            if val is None:
                continue
            v = normalize_hinglish(str(val)).strip()
            v = " ".join(v.split()).strip()
            if not v:
                continue

            # encode field value without EOS (we want exact subseq)
            val_ids = self.tok.encode(v, add_bos=False, add_eos=False, max_length=self.max_length)

            pos = _find_subsequence(raw_core, val_ids)
            if pos is None:
                continue

            start_t[fi] = pos
            end_t[fi] = pos + len(val_ids) - 1
            fmask[fi] = 1

        # MLM corruption uses the padded sequence (so model always sees fixed T)
        mlm_input, mlm_labels = _make_mlm_inputs(
            raw_ids,
            pad_id=self.tok.pad_id,
            mask_id=self.mask_id,
            vocab_size=self.tok.vocab_size,
            rng=self.rng,
            p_mask=self.mlm_prob,
        )

        return TaskCExample(
            input_ids=raw_ids,
            attention_mask=attn,
            start_targets=start_t,
            end_targets=end_t,
            field_mask=fmask,
            mlm_input_ids=mlm_input,
            mlm_labels=mlm_labels,
        )


def as_tensors(ex: TaskCExample) -> dict:
    return {
        "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
        "start_targets": torch.tensor(ex.start_targets, dtype=torch.long),
        "end_targets": torch.tensor(ex.end_targets, dtype=torch.long),
        "field_mask": torch.tensor(ex.field_mask, dtype=torch.long),
        "mlm_input_ids": torch.tensor(ex.mlm_input_ids, dtype=torch.long),
        "mlm_labels": torch.tensor(ex.mlm_labels, dtype=torch.long),
    }
