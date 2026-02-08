from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from opa.models.task_a.model import TinyTransformerLMForTaskA
from opa.tokenization.tokenizer import SentencePieceTokenizer
from opa.utils.text_norm import normalize_hinglish


@dataclass
class TaskAPrediction:
    intent: str
    slots: Dict[str, str]


class TaskARuntime:
    """
    Loads a trained Task A model checkpoint and provides:
      - predict(text) -> intent + slots
    """

    def __init__(self, ckpt_path: Path, device: Optional[str] = None):
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.label_maps = ckpt["label_maps"]
        self.id2intent = {int(k): v for k, v in self.label_maps["id2intent"].items()} if isinstance(list(self.label_maps["id2intent"].keys())[0], str) else self.label_maps["id2intent"]
        # label_maps.json stored id2intent/id2slot with int keys serialized as strings sometimes
        if isinstance(next(iter(self.label_maps["id2slot"].keys())), str):
            self.id2slot = {int(k): v for k, v in self.label_maps["id2slot"].items()}
        else:
            self.id2slot = self.label_maps["id2slot"]

        sp_model_path = Path(ckpt["sp_model_path"])
        self.tokenizer = SentencePieceTokenizer(sp_model_path)

        cfg = ckpt["model_config"]
        self.max_len = int(ckpt["max_len"])

        # Build model and load weights
        self.model = TinyTransformerLMForTaskA(
            vocab_size=self.tokenizer.vocab_size,
            num_intents=len(self.label_maps["intent2id"]),
            num_slot_labels=len(self.label_maps["slot2id"]),
            pad_id=self.tokenizer.pad_id,
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            d_ff=int(cfg["d_ff"]),
            dropout=float(cfg["dropout"]),
            max_len=int(cfg["max_len"]),
            tie_mlm_head=bool(cfg.get("tie_mlm_head", True)),
        )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

        # Device
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, text: str) -> TaskAPrediction:
        t = normalize_hinglish(text)
        ids = self.tokenizer.encode(t, add_bos=False, add_eos=True, max_length=self.max_len)
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids = ids + [self.tokenizer.pad_id] * pad_len
            attn = attn + [0] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        intent_id = int(out.intent_logits.argmax(dim=-1).item())
        intent = self.id2intent[intent_id]

        slot_ids = out.slot_logits.argmax(dim=-1).squeeze(0).tolist()  # [T]
        slot_tags = [self.id2slot[int(i)] for i in slot_ids]

        slots = self._decode_slots(t, slot_tags, attn)

        return TaskAPrediction(intent=intent, slots=slots)

    def _decode_slots(self, text: str, slot_tags: List[str], attn: List[int]) -> Dict[str, str]:
        """
        Convert token-level BIO tags to slot dict.
        We decode token pieces back to text spans by concatenating decoded tokens.
        For Task A (simple slots like next/current/previous, numbers, issue types),
        we can decode values by taking token pieces aligned to a BIO span.
        """
        # Re-tokenize to pieces for span reconstruction
        pieces = self.tokenizer.sp.EncodeAsPieces(text)
        # slot_tags correspond roughly to pieces (+ EOS), but we padded to max_len.
        # We'll only consider up to len(pieces) positions.
        n = min(len(pieces), len(slot_tags))

        spans: Dict[str, List[str]] = {}
        active_key: Optional[str] = None

        def clean_piece(p: str) -> str:
            # SentencePiece uses ▁ to mark word boundary
            s = p.replace("▁", " ").strip()
            return s

        for i in range(n):
            if attn[i] == 0:
                break
            tag = slot_tags[i]
            piece = clean_piece(pieces[i])

            if tag == "O":
                active_key = None
                continue

            if tag.startswith("B-"):
                active_key = tag[2:]
                spans.setdefault(active_key, [])
                if piece:
                    spans[active_key].append(piece)
            elif tag.startswith("I-"):
                key = tag[2:]
                # only continue if same slot is active
                if active_key != key:
                    active_key = key
                    spans.setdefault(active_key, [])
                if piece:
                    spans[active_key].append(piece)

        # post-process into slot dict
        out: Dict[str, str] = {}
        for k, parts in spans.items():
            val = " ".join(parts).strip()
            val = " ".join(val.split())
            if val:
                out[k] = val

        # Optional normalization for known slots
        if "delay_min" in out:
            # keep only digits
            digits = "".join([c for c in out["delay_min"] if c.isdigit()])
            if digits:
                out["delay_min"] = digits
        if "order" in out:
            v = out["order"].lower()
            if "next" in v:
                out["order"] = "next"
            elif "prev" in v or "previous" in v:
                out["order"] = "previous"
            elif "current" in v or "this" in v:
                out["order"] = "current"

        if "issue" in out:
            v = out["issue"].lower()
            # map free text to allowed issue set
            if "missing" in v:
                out["issue"] = "item_missing"
            elif "delay" in v or "late" in v:
                out["issue"] = "restaurant_delay"
            elif "address" in v or "location" in v:
                out["issue"] = "wrong_address"
            elif "pay" in v or "cash" in v or "upi" in v:
                out["issue"] = "payment"
            else:
                out["issue"] = "other"

        return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_a/best.pt")
    ap.add_argument("--text", type=str, required=True)
    args = ap.parse_args()

    rt = TaskARuntime(Path(args.ckpt))
    pred = rt.predict(args.text)
    print(json.dumps({"intent": pred.intent, "slots": pred.slots}, ensure_ascii=False))


if __name__ == "__main__":
    main()
