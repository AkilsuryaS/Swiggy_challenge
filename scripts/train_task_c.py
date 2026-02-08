from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opa.data.dataset_c import TaskCDataset
from opa.data.collate import collate_task_c
from opa.models.task_c.model import TinyEncoderForTaskC
from opa.models.task_c.losses import TaskCLoss
from opa.training.trainer import get_device


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/task_c")
    ap.add_argument("--sp_model", type=str, default="models/tokenizer/task_a/spm.model")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--prefer_mps", action="store_true")
    ap.add_argument("--out_dir", type=str, default="models/task_c")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    sp_model = Path(args.sp_model)

    train_ds = TaskCDataset(
        jsonl_path=data_dir / "train.jsonl",
        sp_model_path=sp_model,
        max_length=args.max_len,
        seed=1337,
        mlm_prob=0.15,
    )
    val_ds = TaskCDataset(
        jsonl_path=data_dir / "val.jsonl",
        sp_model_path=sp_model,
        max_length=args.max_len,
        seed=2020,
        mlm_prob=0.15,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_task_c)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_task_c)

    device = get_device(prefer_mps=args.prefer_mps)
    print(f"Using device: {device}")

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model))

    model = TinyEncoderForTaskC(
        vocab_size=sp.GetPieceSize(),
        pad_id=sp.pad_id(),
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=768,
        dropout=0.1,
        max_len=args.max_len,
        tie_mlm_head=True,
    ).to(device)

    print(f"Model params: {count_params(model):,d}")

    loss_fn = TaskCLoss(pad_id=sp.pad_id(), mlm_weight=0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = 1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            start_t = batch["start_targets"].to(device)
            end_t = batch["end_targets"].to(device)
            fmask = batch["field_mask"].to(device)
            mlm_in = batch["mlm_input_ids"].to(device)
            mlm_lab = batch["mlm_labels"].to(device)

            out = model(input_ids=input_ids, attention_mask=attn, mlm_input_ids=mlm_in)

            loss = loss_fn(
                start_logits=out.start_logits,
                end_logits=out.end_logits,
                start_targets=start_t,
                end_targets=end_t,
                field_mask=fmask,
                mlm_logits=out.mlm_logits,
                mlm_labels=mlm_lab,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = int(input_ids.size(0))
            total += float(loss.detach().cpu()) * bs
            n += bs

        train_loss = total / max(n, 1)

        # val
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            vn = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                start_t = batch["start_targets"].to(device)
                end_t = batch["end_targets"].to(device)
                fmask = batch["field_mask"].to(device)
                mlm_in = batch["mlm_input_ids"].to(device)
                mlm_lab = batch["mlm_labels"].to(device)

                out = model(input_ids=input_ids, attention_mask=attn, mlm_input_ids=mlm_in)
                loss = loss_fn(
                    start_logits=out.start_logits,
                    end_logits=out.end_logits,
                    start_targets=start_t,
                    end_targets=end_t,
                    field_mask=fmask,
                    mlm_logits=out.mlm_logits,
                    mlm_labels=mlm_lab,
                )

                bs = int(input_ids.size(0))
                vtotal += float(loss.detach().cpu()) * bs
                vn += bs

            val_loss = vtotal / max(vn, 1)

        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            ckpt = {
                "model_state": model.state_dict(),
                "sp_model_path": str(sp_model),
                "max_len": int(args.max_len),
                "fields": list(__import__("opa.data.dataset_c", fromlist=["FIELDS"]).FIELDS),
                "model_config": {
                    "d_model": 256,
                    "n_heads": 4,
                    "n_layers": 4,
                    "d_ff": 768,
                    "dropout": 0.1,
                    "max_len": int(args.max_len),
                    "tie_mlm_head": True,
                },
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"✅ saved best checkpoint to {out_dir / 'best.pt'}")

    print(f"Done. Best val_loss={best_val:.4f}")


if __name__ == "__main__":
    main()
