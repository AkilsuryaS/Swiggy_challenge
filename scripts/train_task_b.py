from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opa.data.dataset_b import TaskBDataset
from opa.data.collate import collate_task_b
from opa.models.task_b.model import TinyCausalTransformerLM
from opa.models.task_b.losses import CausalLMLoss
from opa.training.trainer import get_device


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/task_b")
    ap.add_argument("--sp_model", type=str, default="models/tokenizer/task_b/spm.model")
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--prefer_mps", action="store_true")
    ap.add_argument("--out_dir", type=str, default="models/task_b")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    sp_model = Path(args.sp_model)

    train_ds = TaskBDataset(jsonl_path=data_dir / "train.jsonl", sp_model_path=sp_model, max_length=args.max_len)
    val_ds = TaskBDataset(jsonl_path=data_dir / "val.jsonl", sp_model_path=sp_model, max_length=args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_task_b)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_task_b)

    device = get_device(prefer_mps=args.prefer_mps)
    print(f"Using device: {device}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model))

    model = TinyCausalTransformerLM(
        vocab_size=sp.GetPieceSize(),
        pad_id=sp.pad_id(),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=768,
        dropout=0.1,
        max_len=args.max_len,
        tie_lm_head=True,
    ).to(device)

    print(f"Model params: {count_params(model):,d}")

    loss_fn = CausalLMLoss(pad_id=sp.pad_id())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn)
            loss = loss_fn(out.logits, input_ids, loss_mask)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(input_ids.size(0))
            total_loss += float(loss.detach().cpu()) * bs
            n += bs

        train_loss = total_loss / max(n, 1)

        model.eval()
        with torch.no_grad():
            vloss = 0.0
            vn = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                loss_mask = batch["loss_mask"].to(device)
                out = model(input_ids=input_ids, attention_mask=attn)
                loss = loss_fn(out.logits, input_ids, loss_mask)
                bs = int(input_ids.size(0))
                vloss += float(loss.detach().cpu()) * bs
                vn += bs
            val_loss = vloss / max(vn, 1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "sp_model_path": str(sp_model),
                "max_len": int(args.max_len),
                "model_config": {
                    "d_model": 256,
                    "n_heads": 4,
                    "n_layers": 4,
                    "d_ff": 768,
                    "dropout": 0.1,
                    "max_len": int(args.max_len),
                    "tie_lm_head": True,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"✅ saved best checkpoint to {out_dir / 'best.pt'}")

    print(f"Done. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
