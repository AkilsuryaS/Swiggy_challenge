from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class LossBreakdown:
    total: torch.Tensor
    intent: torch.Tensor
    slots: torch.Tensor
    mlm: torch.Tensor


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    pad_id: int,
    unk_id: int,
    eos_id: Optional[int] = None,
    mask_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masked inputs and MLM labels.
    - We DON'T have an explicit <mask> token in SentencePiece; we use UNK as a masking token.
    - Labels are original token ids where masked, else -100.

    Returns:
      mlm_input_ids: [B, T]
      mlm_labels:    [B, T] with -100 for non-masked positions
    """
    device = input_ids.device
    B, T = input_ids.shape

    # candidates: non-pad tokens
    candidates = attention_mask.bool() & (input_ids != pad_id)
    if eos_id is not None:
        candidates = candidates & (input_ids != eos_id)

    # sample mask positions
    rand = torch.rand((B, T), device=device)
    mask_pos = (rand < mask_prob) & candidates

    mlm_labels = torch.full_like(input_ids, fill_value=-100)
    mlm_labels[mask_pos] = input_ids[mask_pos]

    mlm_input_ids = input_ids.clone()
    mlm_input_ids[mask_pos] = int(unk_id)  # mask with UNK

    return mlm_input_ids, mlm_labels


class TaskALoss(nn.Module):
    def __init__(self, *, slot_ignore_index: int = -100):
        super().__init__()
        self.intent_ce = nn.CrossEntropyLoss()
        self.slot_ce = nn.CrossEntropyLoss(ignore_index=slot_ignore_index)
        self.mlm_ce = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        *,
        intent_logits: torch.Tensor,    # [B, num_intents]
        intent_targets: torch.Tensor,   # [B]
        slot_logits: torch.Tensor,      # [B, T, num_slots]
        slot_targets: torch.Tensor,     # [B, T] (pad positions should be -100)
        mlm_logits: torch.Tensor,       # [B, T, vocab]
        mlm_targets: torch.Tensor,      # [B, T] with -100 non-masked
        alpha_mlm: float = 0.3,
    ) -> LossBreakdown:
        li = self.intent_ce(intent_logits, intent_targets)

        B, T, C = slot_logits.shape
        ls = self.slot_ce(slot_logits.view(B * T, C), slot_targets.view(B * T))

        B2, T2, V = mlm_logits.shape
        lm = self.mlm_ce(mlm_logits.view(B2 * T2, V), mlm_targets.view(B2 * T2))

        total = li + ls + (alpha_mlm * lm)
        return LossBreakdown(total=total, intent=li, slots=ls, mlm=lm)
