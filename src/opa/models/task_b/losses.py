from __future__ import annotations

import torch
import torch.nn as nn


class CausalLMLoss(nn.Module):
    """
    Next-token prediction on reply-only tokens using loss_mask.

    logits at t predict token at t+1.

    loss_mask: [B, T], 1 means this position contributes to loss.
    We apply loss_mask to the target positions (t+1 positions).
    """
    def __init__(self, pad_id: int):
        super().__init__()
        self.pad_id = int(pad_id)
        self.ce = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(self, logits: torch.Tensor, input_ids: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        # shift
        logits = logits[:, :-1, :]               # [B, T-1, V]
        targets = input_ids[:, 1:]               # [B, T-1]
        mask = loss_mask[:, 1:]                  # align with targets

        # ignore non-reply positions by setting target to pad_id (ignored)
        targets = targets.clone()
        targets[mask.eq(0)] = self.pad_id

        B, Tm1, V = logits.shape
        return self.ce(logits.reshape(B * Tm1, V), targets.reshape(B * Tm1))
