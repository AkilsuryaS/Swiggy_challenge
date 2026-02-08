from __future__ import annotations

import argparse
from pathlib import Path

from opa.inference.task_c_runtime import TaskCRuntime


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_c/best.pt")
    ap.add_argument("--raw_address", type=str, required=True)
    ap.add_argument("--confidence", type=float, default=0.30)
    args = ap.parse_args()

    rt = TaskCRuntime(Path(args.ckpt))
    out = rt.predict(args.raw_address, confidence_thresh=args.confidence)

    print("\nRaw address:")
    print(out.raw_address)
    print("\nParsed:")
    for k, v in out.parsed.items():
        print(f"- {k:10s}: {v}")


if __name__ == "__main__":
    main()
