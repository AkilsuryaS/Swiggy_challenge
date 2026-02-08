from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from opa.data.dataset_a import TaskADataset
from opa.data.collate import collate_task_a
from opa.models.task_a.model import TinyTransformerLMForTaskA
from opa.models.task_a.losses import TaskALoss
from opa.training.trainer import evaluate, get_device, train_one_epoch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def print_sample_predictions(
    *,
    model: torch.nn.Module,
    device: torch.device,
    sp_model_path: Path,
    label_maps: dict,
    jsonl_path: Path,
    max_len: int,
    k: int = 8,
) -> None:
    """
    Prints a few intent predictions for quick sanity-checking.
    """
    import sentencepiece as spm

    intent2id = {k: int(v) for k, v in label_maps["intent2id"].items()}
    id2intent = {int(v): k for k, v in intent2id.items()}

    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_path))
    pad_id = sp.pad_id()
    eos_id = sp.eos_id()

    rows = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print(f"\n(No rows found in {jsonl_path})")
        return

    rng = random.Random(1337)
    sample = rows if len(rows) <= k else rng.sample(rows, k=k)

    print(f"\n### Sample predictions ({jsonl_path})")
    for i, obj in enumerate(sample, start=1):
        text = str(obj["text"])
        true_intent = str(obj["intent"])
        true_id = int(intent2id[true_intent])

        ids = sp.EncodeAsIds(text) + [eos_id]
        ids = ids[:max_len]
        attn = [1] * len(ids)
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids = ids + [pad_id] * pad_len
            attn = attn + [0] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred_id = int(out.intent_logits.argmax(dim=-1).item())

        ok = "OK" if pred_id == true_id else "WRONG"
        pred_intent = id2intent.get(pred_id, str(pred_id))

        print(f"{i:02d}. [{ok}] true={true_intent:20s} pred={pred_intent:20s} text={text}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/processed/task_a")
    ap.add_argument("--sp_model", type=str, default="models/tokenizer/task_a/spm.model")
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha_mlm", type=float, default=0.3)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--out_dir", type=str, default="models/task_a")
    ap.add_argument("--prefer_mps", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    label_maps_path = data_dir / "label_maps.json"
    maps = json.loads(label_maps_path.read_text(encoding="utf-8"))
    num_intents = len(maps["intent2id"])
    num_slots = len(maps["slot2id"])

    sp_model_path = Path(args.sp_model)

    train_ds = TaskADataset(
        jsonl_path=data_dir / "train.jsonl",
        label_maps_path=label_maps_path,
        sp_model_path=sp_model_path,
        max_length=args.max_len,
    )
    val_ds = TaskADataset(
        jsonl_path=data_dir / "val.jsonl",
        label_maps_path=label_maps_path,
        sp_model_path=sp_model_path,
        max_length=args.max_len,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_task_a)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_task_a)

    device = get_device(prefer_mps=args.prefer_mps)
    print(f"Using device: {device}")

    # Load tokenizer ids (pad/unk/eos) safely
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(str(sp_model_path))
    pad_id = sp.pad_id()
    unk_id = sp.unk_id()
    eos_id = sp.eos_id()

    model = TinyTransformerLMForTaskA(
        vocab_size=sp.GetPieceSize(),
        num_intents=num_intents,
        num_slot_labels=num_slots,
        pad_id=pad_id,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=768,
        dropout=0.1,
        max_len=args.max_len,
        tie_mlm_head=True,
    ).to(device)

    print(f"Model params: {count_params(model):,d}")

    loss_fn = TaskALoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    best_acc = -1.0
    best_epoch = -1
    best_ckpt_path = Path(args.out_dir) / "best.pt"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            train_loader=train_loader,
            device=device,
            pad_id=pad_id,
            unk_id=unk_id,
            eos_id=eos_id,
            alpha_mlm=float(args.alpha_mlm),
            mask_prob=float(args.mask_prob),
        )
        val_metrics = evaluate(model=model, val_loader=val_loader, device=device)

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  train loss: {train_metrics['loss']:.4f} (intent {train_metrics['intent']:.4f}, slots {train_metrics['slots']:.4f}, mlm {train_metrics['mlm']:.4f})")
        print(f"  val intent_acc: {val_metrics['intent_acc']:.4f}")

        if val_metrics["intent_acc"] > best_acc:
            best_acc = val_metrics["intent_acc"]
            best_epoch = int(epoch)
            ckpt = {
                "model_state": model.state_dict(),
                "label_maps": maps,
                "sp_model_path": str(sp_model_path),
                "max_len": int(args.max_len),
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
            torch.save(ckpt, best_ckpt_path)
            print(f"  ✅ saved best checkpoint to {best_ckpt_path}")

    print("\n### Training summary")
    print(f"Best val intent_acc: {best_acc:.4f} (epoch {best_epoch})")
    print(f"Best checkpoint: {best_ckpt_path}")

    # Print a few qualitative predictions from the best checkpoint
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        print_sample_predictions(
            model=model,
            device=device,
            sp_model_path=sp_model_path,
            label_maps=maps,
            jsonl_path=data_dir / "val.jsonl",
            max_len=int(args.max_len),
            k=8,
        )


if __name__ == "__main__":
    main()
