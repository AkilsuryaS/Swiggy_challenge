from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TaskAOutputs:
    intent_logits: torch.Tensor          # [B, num_intents]
    slot_logits: torch.Tensor            # [B, T, num_slots]
    mlm_logits: Optional[torch.Tensor]   # [B, T, vocab] or None


class TinyTransformerLMForTaskA(nn.Module):
    """
    Tiny Transformer encoder trained from scratch.
    Acts as a "language model backbone" via auxiliary MLM loss.

    Heads:
      - intent classification head (pooled)
      - slot tagging head (per token)
      - MLM head (per token vocab logits)

    Export-friendly: pure PyTorch ops.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        num_intents: int,
        num_slot_labels: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 768,
        dropout: float = 0.1,
        max_len: int = 64,
        tie_mlm_head: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.num_intents = int(num_intents)
        self.num_slot_labels = int(num_slot_labels)
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)
        self.max_len = int(max_len)

        self.tok_emb = nn.Embedding(self.vocab_size, self.d_model, padding_idx=self.pad_id)
        self.pos_emb = nn.Embedding(self.max_len, self.d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(d_ff),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)

        # Heads
        self.intent_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.num_intents),
        )
        self.slot_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.num_slot_labels),
        )

        self.mlm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie_mlm_head:
            # weight tying improves LM quality + reduces params
            self.mlm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,   # [B, T] 1 for real tokens, 0 for pad
        mlm_input_ids: Optional[torch.Tensor] = None,  # [B, T] masked inputs (optional)
    ) -> TaskAOutputs:
        """
        If mlm_input_ids is provided, MLM logits are computed using those ids.
        Otherwise MLM logits are computed using input_ids (still valid, but less useful).
        """
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"input length {T} > max_len {self.max_len}. Increase max_len or truncate earlier.")

        device = input_ids.device
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        # encoder uses key_padding_mask: True for pads
        key_padding_mask = attention_mask.eq(0)  # [B, T]

        # main encoding (for intent+slots)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)  # [B, T, D]

        # pooled representation: mean over non-pad tokens
        attn = attention_mask.unsqueeze(-1).float()  # [B, T, 1]
        denom = attn.sum(dim=1).clamp(min=1.0)       # [B, 1]
        pooled = (x * attn).sum(dim=1) / denom       # [B, D]

        intent_logits = self.intent_head(pooled)     # [B, num_intents]
        slot_logits = self.slot_head(x)              # [B, T, num_slot_labels]

        # MLM path (optional masked ids)
        if mlm_input_ids is None:
            mlm_ids = input_ids
        else:
            mlm_ids = mlm_input_ids

        xm = self.tok_emb(mlm_ids) + self.pos_emb(pos)
        xm = self.encoder(xm, src_key_padding_mask=key_padding_mask)
        xm = self.norm(xm)
        mlm_logits = self.mlm_head(xm)               # [B, T, vocab]

        return TaskAOutputs(
            intent_logits=intent_logits,
            slot_logits=slot_logits,
            mlm_logits=mlm_logits,
        )
