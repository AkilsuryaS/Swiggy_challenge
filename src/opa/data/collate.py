from __future__ import annotations

from typing import Dict, List

import torch

from opa.data.dataset_a import TaskAExample, as_tensors


def collate_task_a(batch: List[TaskAExample]) -> Dict[str, torch.Tensor]:
    """
    Batch already padded to max_length in dataset.
    Just stack tensors.
    """
    t = [as_tensors(b) for b in batch]
    out = {}
    for k in t[0].keys():
        out[k] = torch.stack([x[k] for x in t], dim=0)
    return out


from opa.data.dataset_b import TaskBExample, as_tensors as as_tensors_b

def collate_task_b(batch: List[TaskBExample]) -> Dict[str, torch.Tensor]:
    t = [as_tensors_b(b) for b in batch]
    out = {}
    for k in t[0].keys():
        out[k] = torch.stack([x[k] for x in t], dim=0)
    return out
