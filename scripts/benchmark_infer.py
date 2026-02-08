from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from opa.export.quantize import quantize_onnx_dynamic
from opa.export.validate_export import run_onnx_once


def sizeof_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True, help="Path to exported ONNX model (fp32)")
    ap.add_argument("--out_dir", type=str, default="models/task_a/onnx")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--no_quantize", action="store_true", help="Skip quantization and benchmark fp32 only")
    args = ap.parse_args()

    onnx_fp32 = Path(args.onnx)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not onnx_fp32.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_fp32}")

    # Decide which model to benchmark
    if args.no_quantize:
        model_path = onnx_fp32
        print(f"Benchmarking FP32 model: {model_path} ({sizeof_mb(model_path):.2f} MB)")
    else:
        onnx_int8 = out_dir / "task_a_intent_slot.int8.onnx"
        model_path = quantize_onnx_dynamic(onnx_fp32, onnx_int8)
        print(f"✅ Quantized model saved: {model_path}")
        print(f"FP32 size: {sizeof_mb(onnx_fp32):.2f} MB")
        print(f"INT8 size: {sizeof_mb(model_path):.2f} MB")

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
