from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from opa.data.schemas import TaskCRecord
from opa.utils.text_norm import normalize_hinglish


PIN_RE = re.compile(r"\b(\d{6})\b")


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


def norm_field(x):
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    # normalize spacing/casing lightly (keep mostly as-is)
    x = normalize_hinglish(x)
    x = " ".join(x.split())
    return x if x else None


def canonicalize_parsed(parsed: dict) -> dict:
    # enforce key set exactly (no extras)
    keys = ["name","phone","house_flat","building","street","landmark","locality","city","state","pincode"]
    out = {}
    for k in keys:
        out[k] = norm_field(parsed.get(k, None))
    # pincode must be 6 digits or null; if present but not 6 digits -> null
    if out["pincode"] is not None:
        p = out["pincode"]
        if not (len(p) == 6 and p.isdigit()):
            out["pincode"] = None
    return out


def maybe_fill_pincode_from_raw(raw: str, parsed: dict) -> dict:
    """
    Conservative: if parsed.pincode is null but raw has a clear 6-digit pincode, fill it.
    This helps data consistency while still respecting "don't hallucinate".
    """
    if parsed.get("pincode"):
        return parsed
    m = PIN_RE.search(raw)
    if m:
        parsed["pincode"] = m.group(1)
    return parsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--dedupe", action="store_true", help="Deduplicate by normalized raw_address")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    raw_rows = read_jsonl(in_path)

    cleaned: List[dict] = []
    seen = set()
    bad = 0

    for obj in raw_rows:
        try:
            rec = TaskCRecord(**obj)

            raw_addr = normalize_hinglish(rec.raw_address)
            raw_addr = " ".join(raw_addr.split()).strip()

            parsed = canonicalize_parsed(rec.parsed.dict())
            parsed = maybe_fill_pincode_from_raw(raw_addr, parsed)

            if args.dedupe:
                key = raw_addr.lower()
                if key in seen:
                    continue
                seen.add(key)

            # Re-validate after normalization/fill
            _ = TaskCRecord(raw_address=raw_addr, parsed=parsed)

            cleaned.append({"raw_address": raw_addr, "parsed": parsed})
        except Exception:
            bad += 1

    write_jsonl(out_path, cleaned)
    print(f"Input rows:  {len(raw_rows)}")
    print(f"Clean rows:  {len(cleaned)}")
    print(f"Invalid/dropped: {bad}")

    if len(cleaned) < 2500:
        print("\nWARNING: Too few clean samples. Consider regenerating or relaxing schema.\n")


if __name__ == "__main__":
    main()
