from __future__ import annotations

import re


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[“”‘’]")


def normalize_hinglish(text: str) -> str:
    """
    Conservative normalization:
    - lowercase
    - normalize quotes
    - collapse whitespace
    - keep punctuation (useful for tokenization), but remove weird unicode quotes
    """
    t = text.strip().lower()
    t = _PUNCT.sub("'", t)
    t = _WS.sub(" ", t)
    return t


def looks_like_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097F" for ch in text)
