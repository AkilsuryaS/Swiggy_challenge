from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class TrainConfig:
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 50
    device: str = "cpu"


def get_device(prefer_mps: bool = True) -> torch.device:
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    *,
    model,
    loss_fn,
    optimizer,
    train_loader: DataLoader,
    device: torch.device,
    pad_id: int,
    unk_id: int,
    eos_id: Optional[int],
    alpha_mlm: float,
    mask_prob: float,
) -> Dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "intent": 0.0, "slots": 0.0, "mlm": 0.0}
    n = 0

    from opa.models.task_a.losses import make_mlm_batch  # local import to avoid cycles

    pbar = tqdm(train_loader, desc="train", leave=False)
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_id = batch["intent_id"].to(device)
        slot_label_ids = batch["slot_label_ids"].to(device)

        # For slot loss, ignore pad positions
        slot_targets = slot_label_ids.clone()
        slot_targets[attention_mask.eq(0)] = -100

        mlm_input_ids, mlm_targets = make_mlm_batch(
            input_ids, attention_mask, pad_id=pad_id, unk_id=unk_id, eos_id=eos_id, mask_prob=mask_prob
        )

        out = model(input_ids=input_ids, attention_mask=attention_mask, mlm_input_ids=mlm_input_ids)

        loss_bd = loss_fn(
            intent_logits=out.intent_logits,
            intent_targets=intent_id,
            slot_logits=out.slot_logits,
            slot_targets=slot_targets,
            mlm_logits=out.mlm_logits,
            mlm_targets=mlm_targets,
            alpha_mlm=alpha_mlm,
        )

        optimizer.zero_grad(set_to_none=True)
        loss_bd.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        bs = int(input_ids.size(0))
        n += bs
        totals["loss"] += float(loss_bd.total.detach().cpu()) * bs
        totals["intent"] += float(loss_bd.intent.detach().cpu()) * bs
        totals["slots"] += float(loss_bd.slots.detach().cpu()) * bs
        totals["mlm"] += float(loss_bd.mlm.detach().cpu()) * bs

        pbar.set_postfix(loss=totals["loss"] / max(n, 1))

    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    *,
    model,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0

    for batch in tqdm(val_loader, desc="eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_id = batch["intent_id"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = out.intent_logits.argmax(dim=-1)
        correct += int((pred == intent_id).sum().item())
        total += int(intent_id.numel())

    acc = correct / max(total, 1)
    return {"intent_acc": acc}
