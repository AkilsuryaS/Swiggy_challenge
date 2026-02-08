from __future__ import annotations

import numpy as np
import onnxruntime as ort
from pathlib import Path


def run_onnx_once(onnx_path: Path, input_ids: np.ndarray, attention_mask: np.ndarray):
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = sess.run(
        None,
        {"input_ids": input_ids.astype(np.int64), "attention_mask": attention_mask.astype(np.int64)},
    )
    # returns [intent_logits, slot_logits]
    return outputs
