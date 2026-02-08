from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import sentencepiece as spm


@dataclass
class SpmTrainConfig:
    vocab_size: int = 8000
    model_type: str = "unigram"  # "unigram" or "bpe"
    character_coverage: float = 1.0  # roman script, so 1.0 is fine
    # If True, SentencePiece requires it can build exactly `vocab_size` pieces.
    # For small corpora, that can fail (\"Vocabulary size too high\").
    # Setting this to False allows training to succeed with the largest feasible vocab.
    hard_vocab_limit: bool = False
    bos_id: int = 1
    eos_id: int = 2
    pad_id: int = 0
    unk_id: int = 3

    # special tokens (SentencePiece "user_defined_symbols" become single pieces)
    user_defined_symbols: List[str] = None

    def to_json(self) -> str:
        d = asdict(self)
        if d["user_defined_symbols"] is None:
            d["user_defined_symbols"] = []
        return json.dumps(d, indent=2)


def train_sentencepiece(
    *,
    input_text_path: Path,
    out_dir: Path,
    model_prefix: str = "spm",
    cfg: Optional[SpmTrainConfig] = None,
) -> Path:
    """
    Trains a SentencePiece model on a plain-text file (1 sentence per line).
    Returns path to the trained .model.
    """
    cfg = cfg or SpmTrainConfig(user_defined_symbols=["<intent>", "<slots>", "<sep>"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / f"{model_prefix}.model"
    vocab_path = out_dir / f"{model_prefix}.vocab"

    user_syms = cfg.user_defined_symbols or []
    user_syms_str = ",".join(user_syms)

    # SentencePiece trainer accepts one input file with 1 line per sentence
    cmd = (
        f"--input={str(input_text_path)} "
        f"--model_prefix={str(out_dir / model_prefix)} "
        f"--vocab_size={cfg.vocab_size} "
        f"--model_type={cfg.model_type} "
        f"--character_coverage={cfg.character_coverage} "
        f"--hard_vocab_limit={'true' if cfg.hard_vocab_limit else 'false'} "
        f"--pad_id={cfg.pad_id} --bos_id={cfg.bos_id} --eos_id={cfg.eos_id} --unk_id={cfg.unk_id} "
        f"--user_defined_symbols={user_syms_str} "
        f"--byte_fallback=true "  # helps with odd characters/typos robustly
    )

    spm.SentencePieceTrainer.Train(cmd)

    if not model_path.exists() or not vocab_path.exists():
        raise RuntimeError(f"SentencePiece training failed: missing {model_path} or {vocab_path}")

    # Save config for reproducibility
    meta_path = out_dir / "tokenizer_meta.json"
    meta = {
        "input_text_path": str(input_text_path),
        "model_prefix": model_prefix,
        "spm_config": json.loads(cfg.to_json()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return model_path
