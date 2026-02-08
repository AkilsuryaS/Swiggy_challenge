from __future__ import annotations

import argparse
from pathlib import Path

import torch

from opa.models.task_b.model import TinyCausalTransformerLM


class TaskBOnnxWrapper(torch.nn.Module):
    def __init__(self, model: TinyCausalTransformerLM):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # [B, T, vocab]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_b/best.pt")
    ap.add_argument("--out_dir", type=str, default="models/task_b/onnx")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]
    max_len = int(ckpt["max_len"])

    # load tokenizer to get vocab/pad id
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(ckpt["sp_model_path"]))
    vocab_size = sp.GetPieceSize()
    pad_id = sp.pad_id()

    model = TinyCausalTransformerLM(
        vocab_size=vocab_size,
        pad_id=pad_id,
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        d_ff=int(cfg["d_ff"]),
        dropout=0.0,
        max_len=max_len,
        tie_lm_head=bool(cfg.get("tie_lm_head", True)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    wrapper = TaskBOnnxWrapper(model).eval()

    dummy_ids = torch.ones((1, max_len), dtype=torch.long)
    dummy_mask = torch.ones((1, max_len), dtype=torch.long)

    onnx_path = out_dir / "task_b_lm.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=int(args.opset),
        dynamo=False,
    )

    print(f"✅ Exported Task B ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
