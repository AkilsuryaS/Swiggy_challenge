from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialTokens:
    pad: str = "<pad>"
    bos: str = "<s>"
    eos: str = "</s>"
    unk: str = "<unk>"
