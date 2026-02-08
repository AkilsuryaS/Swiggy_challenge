from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from opa.export.quantize import quantize_onnx_dynamic
from opa.export.validate_export import run_onnx_once


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True, help="Path to exported ONNX model")
    ap.add_argument("--out_dir", type=str, default="models/task_a/onnx")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument(
        "--no_quantize",
        action="store_true",
        help="Skip INT8 quantization (useful if quantization fails on this model).",
    )
    args = ap.parse_args()

    onnx_path = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = onnx_path
    if not args.no_quantize:
        q_path = out_dir / "task_a_intent_slot.int8.onnx"
        try:
            quantize_onnx_dynamic(onnx_path, q_path)
            model_path = q_path
            print(f"✅ Quantized: {q_path}")
        except Exception as e:
            # Quantization can fail on some dynamic-shape graphs due to shape inference issues.
            # Fall back to benchmarking the original FP32 model.
            print(f"⚠️ Quantization failed ({type(e).__name__}: {e}). Falling back to FP32 model: {onnx_path}")

    input_ids = np.ones((args.batch, args.max_len), dtype=np.int64)
    attention_mask = np.ones((args.batch, args.max_len), dtype=np.int64)

    # warmup
    for _ in range(args.warmup):
        run_onnx_once(model_path, input_ids, attention_mask)

    # timed
    t0 = time.perf_counter()
    for _ in range(args.iters):
        run_onnx_once(model_path, input_ids, attention_mask)
    t1 = time.perf_counter()

    avg_ms = ((t1 - t0) / args.iters) * 1000.0
    print(f"Avg latency: {avg_ms:.3f} ms  (batch={args.batch}, max_len={args.max_len}, iters={args.iters})")
    print(f"Benchmarked model: {model_path}")


if __name__ == "__main__":
    main()
