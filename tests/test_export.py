from __future__ import annotations

import runpy
import sys
from pathlib import Path

import torch

from opa.export.quantize import quantize_onnx_dynamic
from opa.models.task_c.model import TinyEncoderForTaskC


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _spm_model_path() -> Path:
    return _repo_root() / "models" / "tokenizer" / "task_a" / "spm.model"


def _make_task_c_ckpt(tmp_path: Path) -> Path:
    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(str(_spm_model_path()))

    cfg = {
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "dropout": 0.0,
        "max_len": 16,
        "tie_mlm_head": True,
    }
    model = TinyEncoderForTaskC(
        vocab_size=sp.GetPieceSize(),
        pad_id=sp.pad_id(),
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
        max_len=cfg["max_len"],
        tie_mlm_head=cfg["tie_mlm_head"],
    )

    ckpt = {
        "model_state": model.state_dict(),
        "sp_model_path": str(_spm_model_path()),
        "max_len": cfg["max_len"],
        "model_config": cfg,
    }
    ckpt_path = tmp_path / "task_c_best.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def test_quantize_onnx_dynamic_creates_file(tmp_path: Path):
    class TinyLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=True)

        def forward(self, x: torch.Tensor):
            return self.lin(x)

    model = TinyLinear().eval()
    onnx_fp32 = tmp_path / "lin.onnx"
    onnx_int8 = tmp_path / "lin.int8.onnx"

    dummy = torch.zeros((1, 4), dtype=torch.float32)
    torch.onnx.export(model, dummy, str(onnx_fp32), input_names=["input"], output_names=["output"], opset_version=17)

    out = quantize_onnx_dynamic(onnx_fp32, onnx_int8)
    assert out.exists()


def test_export_task_c_onnx_script(tmp_path: Path):
    ckpt_path = _make_task_c_ckpt(tmp_path)
    out_dir = tmp_path / "onnx"
    script_path = _repo_root() / "scripts" / "export_task_c_onnx.py"

    argv = [str(script_path), "--ckpt", str(ckpt_path), "--out_dir", str(out_dir), "--opset", "17"]
    old_argv = sys.argv
    try:
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        sys.argv = old_argv

    assert (out_dir / "task_c_spans.onnx").exists()
