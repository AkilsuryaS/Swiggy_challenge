from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from opa.data.schemas import TaskARecord


INTENTS = [
    "get_address",
    "call_customer",
    "mark_delivered",
    "mark_picked_up",
    "report_delay",
    "navigation_help",
    "order_issue",
    "customer_unavailable",
]

# Keep slot values in schema.py; here we define slot keys and BIO label inventory.
SLOT_KEYS = ["order", "delay_min", "issue"]


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
                raise RuntimeError(f"Invalid JSON on line {i} in {path}: {e}") from e
    return rows


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def stratified_split(rows: List[dict], seed: int, val_frac: float, test_frac: float) -> Tuple[List[dict], List[dict], List[dict]]:
    rng = random.Random(seed)

    by_intent: Dict[str, List[dict]] = {k: [] for k in INTENTS}
    for r in rows:
        by_intent[r["intent"]].append(r)

    train, val, test = [], [], []
    for intent, items in by_intent.items():
        rng.shuffle(items)
        n = len(items)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        n_train = n - n_val - n_test
        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_label_maps() -> dict:
    """
    Label inventories:
    - intent2id fixed by INTENTS
    - slot tags: BIO for each slot key
      O + (B-<key>, I-<key>) for each key
    """
    intent2id = {k: i for i, k in enumerate(INTENTS)}
    id2intent = {i: k for k, i in intent2id.items()}

    slot_tags = ["O"]
    for k in SLOT_KEYS:
        slot_tags.append(f"B-{k}")
        slot_tags.append(f"I-{k}")

    slot2id = {t: i for i, t in enumerate(slot_tags)}
    id2slot = {i: t for t, i in slot2id.items()}

    return {
        "intent2id": intent2id,
        "id2intent": id2intent,
        "slot2id": slot2id,
        "id2slot": id2slot,
        "slot_keys": SLOT_KEYS,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="task_a", choices=["task_a"])
    ap.add_argument("--clean_jsonl", type=str, required=True, help="data/interim/task_a/clean_v1.jsonl")
    ap.add_argument("--out_dir", type=str, required=True, help="data/processed/task_a")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    in_path = Path(args.clean_jsonl)
    out_dir = Path(args.out_dir)

    raw = read_jsonl(in_path)

    # validate again (cheap safety net)
    rows: List[dict] = []
    intent_counts = Counter()
    for obj in raw:
        rec = TaskARecord(**obj)  # raises if invalid
        rows.append({"text": rec.text, "intent": rec.intent, "slots": rec.slots or {}})
        intent_counts[rec.intent] += 1

    print(f"Loaded clean rows: {len(rows)}")
    print("Intent distribution:")
    for k, v in intent_counts.most_common():
        print(f"  {k:20s} {v:5d}")

    train, val, test = stratified_split(rows, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    write_jsonl(out_dir / "test.jsonl", test)

    label_maps = build_label_maps()
    (out_dir / "label_maps.json").write_text(json.dumps(label_maps, indent=2), encoding="utf-8")

    print(f"\n✅ Wrote splits to: {out_dir}")
    print(f"  train: {len(train)}")
    print(f"  val:   {len(val)}")
    print(f"  test:  {len(test)}")
    print(f"✅ Wrote label maps: {out_dir / 'label_maps.json'}")


if __name__ == "__main__":
    main()
