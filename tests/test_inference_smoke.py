from __future__ import annotations

from pathlib import Path

import torch

from opa.data.dataset_c import FIELDS
from opa.inference.task_c_runtime import TaskCRuntime
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
        "fields": list(FIELDS),
        "model_config": cfg,
    }
    ckpt_path = tmp_path / "task_c_best.pt"
    torch.save(ckpt, ckpt_path)
    return ckpt_path


def test_task_c_runtime_predict_smoke(tmp_path: Path):
    ckpt_path = _make_task_c_ckpt(tmp_path)
    rt = TaskCRuntime(ckpt_path, device="cpu")
    out = rt.predict("flat 2B, MG road, Indore 452001", confidence_thresh=0.2)

    assert out.raw_address
    assert set(out.parsed.keys()) == set(FIELDS)
