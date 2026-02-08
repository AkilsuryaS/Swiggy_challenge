from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from opa.data.schemas import TaskBRecord
from opa.utils.text_norm import normalize_hinglish


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid JSON on line {i}: {e}\nLINE={line[:200]}") from e
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True, help="Path to synthetic Task B JSONL")
    ap.add_argument("--out_path", type=str, required=True, help="Path to write cleaned JSONL")
    ap.add_argument("--dedupe", action="store_true", help="Drop duplicates based on normalized (context, reply)")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    raw = read_jsonl(in_path)

    cleaned: List[dict] = []
    bad = 0
    seen = set()
    length_bins = Counter()

    for obj in raw:
        try:
            rec = TaskBRecord(**obj)
            ctx = normalize_hinglish(rec.context)
            rep = normalize_hinglish(rec.reply)

            if args.dedupe:
                key = (ctx, rep)
                if key in seen:
                    continue
                seen.add(key)

            # Track reply length distribution (words)
            n_words = len(rep.split())
            if n_words <= 4:
                length_bins["<=4"] += 1
            elif n_words <= 8:
                length_bins["5-8"] += 1
            else:
                length_bins["9+"] += 1

            cleaned.append({"context": ctx, "reply": rep})
        except Exception:
            bad += 1

    write_jsonl(out_path, cleaned)

    print(f"Input rows:  {len(raw)}")
    print(f"Clean rows:  {len(cleaned)}")
    print(f"Invalid/dropped: {bad}")
    print("Reply length bins (words):")
    for k, v in length_bins.items():
        print(f"  {k:4s}: {v}")

    if len(cleaned) < 2500:
        print("\nWARNING: Too few clean samples. Consider regenerating or relaxing validation.\n")


if __name__ == "__main__":
    main()
