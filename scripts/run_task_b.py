from __future__ import annotations

import argparse
from pathlib import Path

from opa.inference.task_b_runtime import TaskBRuntime
from opa.models.task_b.decoding import DecodeConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_b/best.pt")
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    rt = TaskBRuntime(Path(args.ckpt))
    gen = rt.generate(
        args.context,
        n_candidates=args.candidates,
        decode_cfg=DecodeConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_on_eos=True,
        ),
    )

    print("\nContext:")
    print(args.context)
    print("\nSmart replies:")
    for i, r in enumerate(gen.replies, start=1):
        print(f"{i}. {r}")


if __name__ == "__main__":
    main()
