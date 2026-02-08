from __future__ import annotations

from pathlib import Path

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def _strip_intermediate_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """
    ONNX shape inference can fail when the model contains incorrect/partial
    intermediate tensor shape annotations (value_info). This happens sometimes
    with exported Transformer graphs.

    To avoid [ShapeInferenceError] conflicts during quantization, we remove
    intermediate value_info entries and keep only inputs/outputs.
    """
    # Remove all intermediate value_info (these are optional)
    del model.graph.value_info[:]
    return model


def quantize_onnx_dynamic(onnx_in: Path, onnx_out: Path) -> Path:
    """
    Dynamic INT8 quantization for CPU inference.

    Fixes common quantization failures by:
      1) Loading the fp32 ONNX
      2) Stripping intermediate shape metadata (value_info)
      3) Saving a cleaned temporary ONNX
      4) Running quantize_dynamic on the cleaned model
    """
    onnx_in = Path(onnx_in)
    onnx_out = Path(onnx_out)
    onnx_out.parent.mkdir(parents=True, exist_ok=True)

    if not onnx_in.exists():
        raise FileNotFoundError(f"ONNX input model not found: {onnx_in}")

    # 1) Load
    model = onnx.load(str(onnx_in))

    # 2) Strip problematic shape annotations
    model = _strip_intermediate_value_info(model)

    # 3) Save cleaned temp model
    cleaned_path = onnx_out.parent / (onnx_in.stem + ".cleaned.onnx")
    onnx.save(model, str(cleaned_path))

    # 4) Quantize cleaned model
    # We explicitly disable shape inference via extra_options as well
    # (some ORT versions still try to infer shapes otherwise).
    quantize_dynamic(
        model_input=str(cleaned_path),
        model_output=str(onnx_out),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul", "Gemm"],
        extra_options={
            "DisableShapeInference": True,
            # Safe knobs for transformer-ish graphs:
            "MatMulConstBOnly": True,
        },
    )

    return onnx_out
