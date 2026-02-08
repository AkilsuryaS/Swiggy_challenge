from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

from opa.tokenization.tokenizer import SentencePieceTokenizer


@dataclass(frozen=True)
class TaskAExample:
    """
    Model-ready example for Task A.

    Notes:
    - `input_ids` and `attention_mask` are already padded to `max_length`.
    - `slot_label_ids` are BIO tags aligned to token positions.
      For now we generate a conservative all-\"O\" sequence (no spans), since
      the current dataset format stores slot *values* but not character spans.
    """

    input_ids: List[int]
    attention_mask: List[int]
    intent_id: int
    slot_label_ids: List[int]


def as_tensors(ex: TaskAExample) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor(ex.input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(ex.attention_mask, dtype=torch.long),
        "intent_id": torch.tensor(ex.intent_id, dtype=torch.long),
        "slot_label_ids": torch.tensor(ex.slot_label_ids, dtype=torch.long),
    }


class TaskADataset(Dataset[TaskAExample]):
    def __init__(
        self,
        *,
        jsonl_path: Path,
        label_maps_path: Path,
        sp_model_path: Path,
        max_length: int = 64,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.label_maps_path = Path(label_maps_path)
        self.sp_model_path = Path(sp_model_path)
        self.max_length = int(max_length)

        maps = json.loads(self.label_maps_path.read_text(encoding="utf-8"))
        self.intent2id: Dict[str, int] = {k: int(v) for k, v in maps["intent2id"].items()}
        self.slot2id: Dict[str, int] = {k: int(v) for k, v in maps["slot2id"].items()}

        if "O" not in self.slot2id:
            raise RuntimeError("label_maps.json missing slot tag 'O'")
        self.o_id = int(self.slot2id["O"])

        self.tok = SentencePieceTokenizer(self.sp_model_path)

        self._examples: List[TaskAExample] = []
        self._load()

    def _load(self) -> None:
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                text = str(obj["text"]).strip()
                intent = str(obj["intent"]).strip()

                if intent not in self.intent2id:
                    raise RuntimeError(f"Unknown intent '{intent}' at {self.jsonl_path}:{line_no}")

                # Encode and pad
                ids = self.tok.encode(text, add_bos=False, add_eos=True, max_length=self.max_length)
                attn = [1] * len(ids)
                if len(ids) < self.max_length:
                    pad_len = self.max_length - len(ids)
                    ids = ids + [self.tok.pad_id] * pad_len
                    attn = attn + [0] * pad_len
                else:
                    ids = ids[: self.max_length]
                    attn = attn[: self.max_length]

                # Slot labels: conservative all-O (no span supervision available)
                slot_labels = [self.o_id] * self.max_length

                self._examples.append(
                    TaskAExample(
                        input_ids=ids,
                        attention_mask=attn,
                        intent_id=int(self.intent2id[intent]),
                        slot_label_ids=slot_labels,
                    )
                )

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> TaskAExample:
        return self._examples[idx]
