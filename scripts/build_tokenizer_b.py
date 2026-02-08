from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from opa.tokenization.spm_train import SpmTrainConfig, train_sentencepiece


def read_task_b_texts(clean_jsonl: Path) -> List[str]:
    """
    Task B rows: {"context": "...", "reply": "..."}
    We train tokenizer on BOTH context and reply (better subwords).
    """
    texts: List[str] = []
    with clean_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ctx = str(obj.get("context", "")).strip()
            rep = str(obj.get("reply", "")).strip()
            if ctx:
                texts.append(ctx)
            if rep:
                texts.append(rep)
    return texts


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln.replace("\n", " ").strip() + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_b_clean_jsonl", type=str, required=True, help="data/interim/task_b/clean_v2.jsonl")
    ap.add_argument("--out_dir", type=str, default="models/tokenizer/task_b")
    ap.add_argument("--vocab_size", type=int, default=4000)
    ap.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"])
    args = ap.parse_args()

    in_path = Path(args.task_b_clean_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    texts = read_task_b_texts(in_path)
    if len(texts) < 100:
        raise RuntimeError(f"Too few texts ({len(texts)}). Check: {in_path}")

    train_txt = out_dir / "spm_train_text.txt"
    write_lines(train_txt, texts)

    # Smaller vocab is usually enough for short smart replies
    cfg = SpmTrainConfig(
        vocab_size=int(args.vocab_size),
        model_type=str(args.model_type),
        character_coverage=1.0,
        user_defined_symbols=["<sep>"],
    )

    model_path = train_sentencepiece(
        input_text_path=train_txt,
        out_dir=out_dir,
        model_prefix="spm",
        cfg=cfg,
    )
    print(f"✅ Task B tokenizer trained: {model_path}")
    print(f"Artifacts in: {out_dir}")


if __name__ == "__main__":
    main()
