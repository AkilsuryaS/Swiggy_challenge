from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_onnx_dynamic(onnx_in: Path, onnx_out: Path) -> Path:
    onnx_in = Path(onnx_in)
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(onnx_in),
        model_output=str(onnx_out),
        weight_type=QuantType.QInt8,
    )
    return onnx_out
