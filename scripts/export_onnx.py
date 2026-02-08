from __future__ import annotations

import argparse
from pathlib import Path

import torch

from opa.models.task_a.model import TinyTransformerLMForTaskA


class TaskAOnnxWrapper(torch.nn.Module):
    """
    Export only what we need for inference:
      - intent_logits
      - slot_logits
    MLM logits are not required at inference time.
    """
    def __init__(self, model: TinyTransformerLMForTaskA):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, mlm_input_ids=None)
        return out.intent_logits, out.slot_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_a/best.pt")
    ap.add_argument("--out_dir", type=str, default="models/task_a/onnx")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["model_config"]

    # Load SentencePiece to get vocab + pad_id
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(ckpt["sp_model_path"]))
    vocab_size = sp.GetPieceSize()
    pad_id = sp.pad_id()

    num_intents = len(ckpt["label_maps"]["intent2id"])
    num_slots = len(ckpt["label_maps"]["slot2id"])
    max_len = int(ckpt["max_len"])

    model = TinyTransformerLMForTaskA(
        vocab_size=vocab_size,
        num_intents=num_intents,
        num_slot_labels=num_slots,
        pad_id=pad_id,
        d_model=int(cfg["d_model"]),
        n_heads=int(cfg["n_heads"]),
        n_layers=int(cfg["n_layers"]),
        d_ff=int(cfg["d_ff"]),
        dropout=0.0,  # export in eval mode
        max_len=max_len,
        tie_mlm_head=bool(cfg.get("tie_mlm_head", True)),
    )
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    wrapper = TaskAOnnxWrapper(model).eval()

    # dummy inputs (dynamic batch, fixed max_len for simplicity)
    input_ids = torch.ones((1, max_len), dtype=torch.long)
    attention_mask = torch.ones((1, max_len), dtype=torch.long)

    onnx_path = out_dir / "task_a_intent_slot.onnx"
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        f=str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch"},
        },
        opset_version=int(args.opset),
    )

    print(f"✅ Exported ONNX: {onnx_path}")


if __name__ == "__main__":
    main()
