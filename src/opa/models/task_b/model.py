from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class TaskBOutputs:
    logits: torch.Tensor  # [B, T, vocab]


class TinyCausalTransformerLM(nn.Module):
    """
    Proper decoder-only (causal) Transformer LM using TransformerEncoder with a causal mask.

    Trained from scratch on:
      "<context> <sep> <reply> </s>"

    Export-friendly, stable.
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
        max_len: int = 96,
        tie_lm_head: bool = True,
    ):
        super().__init__()
        self.vocab_size = int(vocab_size)
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
        self.blocks = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)

        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
        if tie_lm_head:
            self.lm_head.weight = self.tok_emb.weight

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        # float mask: 0 allowed, -inf blocked
        m = torch.full((T, T), float("-inf"), device=device)
        m = torch.triu(m, diagonal=1)
        return m

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TaskBOutputs:
        """
        input_ids: [B, T]
        attention_mask: [B, T] 1 real token, 0 pad
        """
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"input length {T} > max_len {self.max_len}")

        device = input_ids.device
        pos = torch.arange(0, T, device=device).unsqueeze(0).expand(B, T)

        x = self.tok_emb(input_ids) + self.pos_emb(pos)

        # key padding: True where pad
        key_padding_mask = attention_mask.eq(0)  # [B, T]
        causal = self._causal_mask(T, device=device)

        x = self.blocks(x, mask=causal, src_key_padding_mask=key_padding_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # [B, T, vocab]
        return TaskBOutputs(logits=logits)
