from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from opa.data.dataset_c import FIELDS


@dataclass
class TaskCOutputs:
    start_logits: torch.Tensor  # [B, F, T]
    end_logits: torch.Tensor    # [B, F, T]
    mlm_logits: Optional[torch.Tensor] = None  # [B, T, V]


class TinyEncoderForTaskC(nn.Module):
    """
    Transformer encoder trained from scratch.
    Outputs:
      - span heads: start/end per field over tokens
      - MLM head (aux) for "language model" requirement
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pad_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 768,
        dropout: float = 0.1,
        max_len: int = 128,
        tie_mlm_head: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.pad_id = int(pad_id)
        self.d_model = int(d_model)
        self.max_len = int(max_len)
        self.num_fields = len(FIELDS)

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

        # span heads: project token reps -> per-field logits
        self.start_head = nn.Linear(self.d_model, self.num_fields, bias=True)  # [B,T,F]
        self.end_head = nn.Linear(self.d_model, self.num_fields, bias=True)    # [B,T,F]

        # MLM head
        self.mlm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie_mlm_head:
            self.mlm_head.weight = self.tok_emb.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mlm_input_ids: Optional[torch.Tensor] = None,
    ) -> TaskCOutputs:
        """
        input_ids: [B,T]
        attention_mask: [B,T] (1 real, 0 pad)
        mlm_input_ids: optional, if provided compute mlm_logits on that corrupted input
        """
        B, T = input_ids.shape
        device = input_ids.device
        if T > self.max_len:
            raise ValueError(f"T={T} > max_len={self.max_len}")

        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        def encode(x_ids: torch.Tensor) -> torch.Tensor:
            x = self.tok_emb(x_ids) + self.pos_emb(pos)
            key_padding_mask = attention_mask.eq(0)
            x = self.encoder(x, src_key_padding_mask=key_padding_mask)
            x = self.norm(x)
            return x  # [B,T,D]

        h = encode(input_ids)

        # Span logits
        start = self.start_head(h).transpose(1, 2)  # [B,F,T]
        end = self.end_head(h).transpose(1, 2)      # [B,F,T]

        mlm_logits = None
        if mlm_input_ids is not None:
            hm = encode(mlm_input_ids)
            mlm_logits = self.mlm_head(hm)  # [B,T,V]

        return TaskCOutputs(start_logits=start, end_logits=end, mlm_logits=mlm_logits)
