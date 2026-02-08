from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from opa.data.schemas import TaskARecord, TaskBRecord


# -----------------------------
# Task A config
# -----------------------------
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
SLOT_KEYS = ["order", "delay_min", "issue"]


# -----------------------------
# Utils
# -----------------------------
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


def simple_split(rows: List[dict], seed: int, val_frac: float, test_frac: float):
    rng = random.Random(seed)
    rng.shuffle(rows)
    n = len(rows)
    n_test = int(round(n * test_frac))
    n_val = int(round(n * val_frac))
    n_train = n - n_val - n_test
    return (
        rows[:n_train],
        rows[n_train : n_train + n_val],
        rows[n_train + n_val :],
    )


def stratified_split_task_a(rows: List[dict], seed: int, val_frac: float, test_frac: float):
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


def build_label_maps_task_a() -> dict:
    intent2id = {k: i for i, k in enumerate(INTENTS)}
    id2intent = {i: k for k, i in intent2id.items()}

    slot_tags = ["O"]
    for k in SLOT_KEYS:
        slot_tags.extend([f"B-{k}", f"I-{k}"])

    slot2id = {t: i for i, t in enumerate(slot_tags)}
    id2slot = {i: t for t, i in slot2id.items()}

    return {
        "intent2id": intent2id,
        "id2intent": id2intent,
        "slot2id": slot2id,
        "id2slot": id2slot,
        "slot_keys": SLOT_KEYS,
    }


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["task_a", "task_b"])
    ap.add_argument("--clean_jsonl", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.10)
    args = ap.parse_args()

    rows_raw = read_jsonl(Path(args.clean_jsonl))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.task == "task_a":
        rows = []
        intent_counts = Counter()
        for r in rows_raw:
            rec = TaskARecord(**r)
            rows.append({"text": rec.text, "intent": rec.intent, "slots": rec.slots or {}})
            intent_counts[rec.intent] += 1

        print("Intent distribution:")
        for k, v in intent_counts.items():
            print(f"  {k:20s} {v}")

        train, val, test = stratified_split_task_a(
            rows, args.seed, args.val_frac, args.test_frac
        )

        write_jsonl(out_dir / "train.jsonl", train)
        write_jsonl(out_dir / "val.jsonl", val)
        write_jsonl(out_dir / "test.jsonl", test)

        label_maps = build_label_maps_task_a()
        (out_dir / "label_maps.json").write_text(
            json.dumps(label_maps, indent=2), encoding="utf-8"
        )

        print(f"✅ Task A splits written to {out_dir}")

    elif args.task == "task_b":
        rows = []
        for r in rows_raw:
            rec = TaskBRecord(**r)
            rows.append({"context": rec.context, "reply": rec.reply})

        train, val, test = simple_split(
            rows, args.seed, args.val_frac, args.test_frac
        )

        write_jsonl(out_dir / "train.jsonl", train)
        write_jsonl(out_dir / "val.jsonl", val)
        write_jsonl(out_dir / "test.jsonl", test)

        print(f"✅ Task B splits written to {out_dir}")
        print(f"  train: {len(train)}")
        print(f"  val:   {len(val)}")
        print(f"  test:  {len(test)}")


if __name__ == "__main__":
    main()
