from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import sentencepiece as spm


@dataclass
class TokenizerOutput:
    input_ids: List[int]
    attention_mask: List[int]


class SentencePieceTokenizer:
    def __init__(self, model_path: Path):
        self.model_path = Path(model_path)
        self.sp = spm.SentencePieceProcessor()
        ok = self.sp.Load(str(self.model_path))
        if not ok:
            raise RuntimeError(f"Failed to load SentencePiece model: {self.model_path}")

        # ids come from training config in spm_train.py
        self.pad_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.unk_id = self.sp.unk_id()

    @property
    def vocab_size(self) -> int:
        return self.sp.GetPieceSize()

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = True,
        max_length: Optional[int] = None,
    ) -> List[int]:
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]

        if max_length is not None:
            ids = ids[:max_length]
        return ids

    def decode(self, ids: List[int]) -> str:
        # remove padding for readability
        ids = [i for i in ids if i != self.pad_id]
        return self.sp.DecodeIds(ids)

    def encode_batch_pad(
        self,
        texts: List[str],
        *,
        max_length: int,
        add_bos: bool = False,
        add_eos: bool = True,
    ) -> Tuple[List[List[int]], List[List[int]]]:
        batch_ids: List[List[int]] = []
        batch_mask: List[List[int]] = []

        for t in texts:
            ids = self.encode(t, add_bos=add_bos, add_eos=add_eos, max_length=max_length)
            attn = [1] * len(ids)

            # pad
            if len(ids) < max_length:
                pad_len = max_length - len(ids)
                ids = ids + [self.pad_id] * pad_len
                attn = attn + [0] * pad_len

            batch_ids.append(ids)
            batch_mask.append(attn)

        return batch_ids, batch_mask
