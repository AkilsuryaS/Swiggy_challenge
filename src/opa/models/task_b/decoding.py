from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class DecodeConfig:
    max_new_tokens: int = 24
    temperature: float = 0.9
    top_k: int = 30
    stop_on_eos: bool = True


def top_k_sample(logits: torch.Tensor, *, top_k: int, temperature: float) -> int:
    """
    logits: [vocab]
    returns sampled token id
    """
    if temperature <= 0:
        # greedy
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
        probs = torch.softmax(v, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(ix[sampled].item())

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def strip_after_sep(text: str, sep_token: str = "<sep>") -> str:
    # helper if decode includes context; we only want reply portion
    if sep_token in text:
        return text.split(sep_token, 1)[1].strip()
    return text.strip()
