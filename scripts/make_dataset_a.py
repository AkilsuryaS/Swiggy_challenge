from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import List

from opa.data.schemas import TaskARecord
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
    ap.add_argument("--in_path", type=str, required=True, help="Path to synthetic Task A JSONL")
    ap.add_argument("--out_path", type=str, required=True, help="Path to write cleaned JSONL")
    ap.add_argument("--dedupe", action="store_true", help="Drop exact-duplicate normalized texts")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    raw = read_jsonl(in_path)

    cleaned: List[dict] = []
    intent_counts = Counter()
    bad = 0

    seen = set()

    for obj in raw:
        try:
            # validate schema + constraints
            rec = TaskARecord(**obj)
            text_norm = normalize_hinglish(rec.text)

            if args.dedupe:
                key = (text_norm, rec.intent, json.dumps(rec.slots, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)

            cleaned_obj = {
                "text": text_norm,
                "intent": rec.intent,
                "slots": rec.slots or {},
            }
            cleaned.append(cleaned_obj)
            intent_counts[rec.intent] += 1
        except Exception:
            bad += 1

    write_jsonl(out_path, cleaned)

    total = len(cleaned)
    print(f"Input rows: {len(raw)}")
    print(f"Clean rows: {total}")
    print(f"Dropped/invalid: {bad}")
    print("Intent distribution:")
    for k, v in intent_counts.most_common():
        print(f"  {k:20s} {v:5d} ({(v/max(total,1))*100:5.1f}%)")

    # hard expectation for you: around 3000; but allow a little mismatch after cleaning
    if total < 2500:
        print("\nWARNING: Too few clean samples. Consider regenerating or relaxing slot rules.\n")


if __name__ == "__main__":
    main()
