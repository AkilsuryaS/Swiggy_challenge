from __future__ import annotations

import argparse
from pathlib import Path

import torch

from opa.models.task_c.model import TinyEncoderForTaskC


class TaskCOnnxWrapper(torch.nn.Module):
    def __init__(self, model: TinyEncoderForTaskC):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, mlm_input_ids=None)
        return out.start_logits, out.end_logits  # [B,F,T], [B,F,T]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_c/best.pt")
    ap.add_argument("--out_dir", type=str, default="models/task_c/onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    max_len = int(ckpt["max_len"])

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(ckpt["sp_model_path"]))
    vocab_size = sp.GetPieceSize()
    pad_id = sp.pad_id()

    model = TinyEncoderForTaskC(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        d_ff=int(cfg["d_ff"]),
        dropout=0.0,
        max_len=max_len,
        tie_mlm_head=bool(cfg.get("tie_mlm_head", True)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    wrapper = TaskCOnnxWrapper(model).eval()

    dummy_ids = torch.ones((1, max_len), dtype=torch.long)
    dummy_mask = torch.ones((1, max_len), dtype=torch.long)

    onnx_path = out_dir / "task_c_spans.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "start_logits": {0: "batch"},
            "end_logits": {0: "batch"},
        },
        opset_version=int(args.opset),
    )

    print(f"✅ Exported Task C ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
