from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from opa.tokenization.tokenizer import SentencePieceTokenizer
from opa.utils.text_norm import normalize_hinglish


@dataclass
class TaskBExample:
    input_ids: List[int]
    attention_mask: List[int]
    loss_mask: List[int]  # 1 where we compute loss (reply tokens), 0 elsewhere


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class TaskBDataset(Dataset):
    """
    Builds: "<context> <sep> <reply> </s>"

    loss_mask=0 for context + <sep>
    loss_mask=1 for reply tokens (and EOS)
    """
    def __init__(
        self,
        *,
        jsonl_path: Path,
        sp_model_path: Path,
        max_length: int = 96,
        sep_token: str = "<sep>",
    ):
        self.rows = _read_jsonl(Path(jsonl_path))
        self.tok = SentencePieceTokenizer(Path(sp_model_path))
        self.max_length = int(max_length)
        self.sep_token = sep_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> TaskBExample:
        r = self.rows[idx]
        ctx = normalize_hinglish(r["context"])
        rep = normalize_hinglish(r["reply"])

        prompt = f"{ctx} {self.sep_token}".strip()
        full = f"{ctx} {self.sep_token} {rep}".strip()

        # Encode prompt WITHOUT eos (we want reply loss after prompt)
        prompt_ids = self.tok.encode(prompt, add_bos=False, add_eos=False, max_length=self.max_length)

        # Encode full WITH eos
        full_ids = self.tok.encode(full, add_bos=False, add_eos=True, max_length=self.max_length)

        # Attention mask
        attn = [1] * len(full_ids)

        # loss_mask: 0 for prompt part, 1 for reply part
        lm = [0] * len(full_ids)
        start = min(len(prompt_ids), len(full_ids))
        for i in range(start, len(full_ids)):
            lm[i] = 1

        # Pad
        if len(full_ids) < self.max_length:
            pad_len = self.max_length - len(full_ids)
            full_ids = full_ids + [self.tok.pad_id] * pad_len
            attn = attn + [0] * pad_len
            lm = lm + [0] * pad_len

        return TaskBExample(input_ids=full_ids, attention_mask=attn, loss_mask=lm)


def as_tensors(ex: TaskBExample) -> dict:
    return {
        "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
        "loss_mask": torch.tensor(ex.loss_mask, dtype=torch.long),
    }
