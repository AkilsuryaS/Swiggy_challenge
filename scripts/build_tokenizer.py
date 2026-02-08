from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import List, Tuple

from opa.tokenization.spm_train import SpmTrainConfig, train_sentencepiece


def read_texts_from_task_a_jsonl(path: Path) -> List[str]:
    texts: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            txt = str(obj.get("text", "")).strip()
            if txt:
                texts.append(txt)
    return texts


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.replace("\n", " ").strip() + "\n")


def estimate_safe_vocab_size(texts: List[str], requested: int) -> Tuple[int, dict]:
    """
    SentencePiece cannot create more pieces than the corpus supports.
    We estimate a conservative upper bound and clamp requested vocab_size.

    Heuristics:
    - Count unique "word-like" tokens (alnum + apostrophe)
    - Cap vocab_size to a fraction of unique tokens + some headroom
    - Also keep a minimum reasonable vocab (e.g. 256)
    """
    token_pat = re.compile(r"[a-zA-Z0-9']+")
    uniq = set()
    total_tokens = 0
    for t in texts:
        toks = token_pat.findall(t.lower())
        total_tokens += len(toks)
        uniq.update(toks)

    uniq_count = max(1, len(uniq))

    # Conservative cap: a bit above unique tokens, but not too high
    # SentencePiece pieces are subwords; still, small corpora limit possible merges.
    # This cap prevents "max was X" errors for small datasets.
    cap = int(max(256, min(requested, uniq_count + 200, uniq_count * 2)))

    stats = {
        "requested_vocab_size": requested,
        "unique_word_tokens": uniq_count,
        "total_word_tokens": total_tokens,
        "clamped_vocab_size": cap,
    }
    return cap, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_a_clean_jsonl", type=str, required=True, help="data/interim/task_a/clean_v1.jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="models/tokenizer/task_a/")
    ap.add_argument("--vocab_size", type=int, default=8000)
    ap.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"])
    args = ap.parse_args()

    in_path = Path(args.task_a_clean_jsonl)
    out_dir = Path(args.out_dir)

    texts = read_texts_from_task_a_jsonl(in_path)
    if len(texts) < 50:
        raise RuntimeError(f"Too few texts ({len(texts)}). Check input file: {in_path}")

    # Write training text (1 sentence per line)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_txt_path = out_dir / "spm_train_text.txt"
    write_lines(train_txt_path, texts)

    # Clamp vocab size to what the corpus can realistically support
    vocab_size, stats = estimate_safe_vocab_size(texts, int(args.vocab_size))
    print("Tokenizer vocab sizing:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    cfg = SpmTrainConfig(
        vocab_size=vocab_size,
        model_type=str(args.model_type),
        character_coverage=1.0,
        user_defined_symbols=["<intent>", "<slots>", "<sep>"],
    )

    # Train (and retry with smaller vocab if SentencePiece still complains)
    try:
        model_path = train_sentencepiece(
            input_text_path=train_txt_path,
            out_dir=out_dir,
            model_prefix="spm",
            cfg=cfg,
        )
    except Exception as e:
        # Fallback: cut vocab further and retry once
        fallback = max(256, int(vocab_size * 0.75))
        print(f"\n⚠️ SentencePiece failed with vocab_size={vocab_size}. Retrying with vocab_size={fallback}...")
        cfg.vocab_size = fallback
        model_path = train_sentencepiece(
            input_text_path=train_txt_path,
            out_dir=out_dir,
            model_prefix="spm",
            cfg=cfg,
        )

    print(f"\n✅ Tokenizer trained: {model_path}")
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
