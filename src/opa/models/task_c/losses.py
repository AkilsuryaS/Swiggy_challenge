from __future__ import annotations

import torch
import torch.nn as nn


class TaskCLoss(nn.Module):
    """
    Span loss (start/end) per field + MLM auxiliary loss.

    start_logits/end_logits: [B,F,T]
    start_targets/end_targets: [B,F] (token index)
    field_mask: [B,F] (1=present)
    mlm_logits: [B,T,V]
    mlm_labels: [B,T] (token id or pad_id(ignore))
    """
    def __init__(self, pad_id: int, mlm_weight: float = 0.3):
        super().__init__()
        self.pad_id = int(pad_id)
        self.mlm_weight = float(mlm_weight)
        self.ce_tok = nn.CrossEntropyLoss(reduction="none")  # we mask manually
        self.ce_mlm = nn.CrossEntropyLoss(ignore_index=self.pad_id)

    def forward(
        self,
        *,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_targets: torch.Tensor,
        end_targets: torch.Tensor,
        field_mask: torch.Tensor,
        mlm_logits: torch.Tensor,
        mlm_labels: torch.Tensor,
    ) -> torch.Tensor:
        B, F, T = start_logits.shape

        # span losses per field
        # reshape to [B*F, T]
        s = start_logits.reshape(B * F, T)
        e = end_logits.reshape(B * F, T)
        st = start_targets.reshape(B * F)
        et = end_targets.reshape(B * F)
        fm = field_mask.reshape(B * F).float()  # 0/1

        s_loss = self.ce_tok(s, st)  # [B*F]
        e_loss = self.ce_tok(e, et)  # [B*F]
        span_loss = (s_loss + e_loss) * fm

        denom = fm.sum().clamp(min=1.0)
        span_loss = span_loss.sum() / denom

        # MLM loss
        mlm = self.ce_mlm(mlm_logits.reshape(-1, mlm_logits.size(-1)), mlm_labels.reshape(-1))

        return span_loss + self.mlm_weight * mlm
