from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from opa.models.task_c.model import TinyEncoderForTaskC
from opa.tokenization.tokenizer import SentencePieceTokenizer
from opa.utils.text_norm import normalize_hinglish


@dataclass
class TaskCParsedOut:
    raw_address: str
    parsed: Dict[str, Optional[str]]


def _valid_len(attn: torch.Tensor) -> int:
    # attn: [T] int64 {0,1}
    return int(attn.sum().item())


def _extract_span_text(tok: SentencePieceTokenizer, ids: List[int], start: int, end: int) -> str:
    if start < 0 or end < start or start >= len(ids):
        return ""
    end = min(end, len(ids) - 1)
    span_ids = ids[start : end + 1]
    text = tok.decode(span_ids)
    text = " ".join(text.split()).strip()
    return text


class TaskCRuntime:
    def __init__(self, ckpt_path: Path, device: Optional[str] = None):
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.fields: List[str] = list(ckpt["fields"])
        self.max_len = int(ckpt["max_len"])
        self.sp_model_path = Path(ckpt["sp_model_path"])
        self.tok = SentencePieceTokenizer(self.sp_model_path)

        cfg = ckpt["model_config"]
        self.model = TinyEncoderForTaskC(
            vocab_size=self.tok.vocab_size,
            pad_id=self.tok.pad_id,
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            d_ff=int(cfg["d_ff"]),
            dropout=0.0,
            max_len=self.max_len,
            tie_mlm_head=bool(cfg.get("tie_mlm_head", True)),
        )
        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.eval()

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
    def predict(
        self,
        raw_address: str,
        *,
        confidence_thresh: float = 0.30,
    ) -> TaskCParsedOut:
        raw = normalize_hinglish(raw_address)
        raw = " ".join(raw.split()).strip()

        ids = self.tok.encode(raw, add_bos=False, add_eos=True, max_length=self.max_len)
        attn = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_len = self.max_len - len(ids)
            ids = ids + [self.tok.pad_id] * pad_len
            attn = attn + [0] * pad_len

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

        out = self.model(input_ids=input_ids, attention_mask=attention_mask, mlm_input_ids=None)
        start_logits = out.start_logits[0]
        end_logits = out.end_logits[0]

        T_valid = _valid_len(attention_mask[0])
        ids_core = ids[:T_valid]

        parsed: Dict[str, Optional[str]] = {f: None for f in self.fields}

        # field-specific thresholds (tune lightly)
        field_thresh = {f: confidence_thresh for f in self.fields}
        field_thresh["name"] = max(confidence_thresh, 0.55)
        field_thresh["phone"] = max(confidence_thresh, 0.65)
        field_thresh["city"] = max(confidence_thresh, 0.40)
        field_thresh["state"] = max(confidence_thresh, 0.40)

        bad_name_tokens = {
            "flat","house","h","no","hno","road","rd","street","st","near","opp","behind",
            "tower","apts","apartment","society","phase","block","wing","mg","main"
        }

        def looks_like_masked_phone(s: str) -> bool:
            s2 = s.replace(" ", "")
            # allow masked like 98xxxxxx12 or 9xxxxxxx21
            if "x" in s2.lower():
                return True
            digits = "".join([c for c in s2 if c.isdigit()])
            # if it has 10 digits, it's likely a real phone in raw -> we actually want null per constraints
            # so we only accept if it's short or masked
            if len(digits) >= 10:
                return False
            return False

        for fi, field in enumerate(self.fields):
            s = start_logits[fi, :T_valid]
            e = end_logits[fi, :T_valid]

            ps = F.softmax(s, dim=-1)
            pe = F.softmax(e, dim=-1)

            s_idx = int(torch.argmax(ps).item())
            e_idx = int(torch.argmax(pe).item())
            conf = float(ps[s_idx].item() * pe[e_idx].item())

            if e_idx < s_idx:
                continue
            if conf < field_thresh.get(field, confidence_thresh):
                continue

            text = _extract_span_text(self.tok, ids_core, s_idx, e_idx)
            if not text:
                continue

            text = " ".join(text.split()).strip()

            # field constraints / cleanup
            if field == "pincode":
                digits = "".join([c for c in text if c.isdigit()])
                text = digits if len(digits) == 6 else ""
                if not text:
                    continue

            if field == "phone":
                if not looks_like_masked_phone(text):
                    continue

            if field == "name":
                # reject if it looks like an address keyword
                low = text.lower()
                toks = [t for t in low.replace(",", " ").split() if t]
                if any(t in bad_name_tokens for t in toks):
                    continue
                # also reject extremely short generic name fragments
                if len(toks) == 1 and len(toks[0]) <= 3:
                    continue

            parsed[field] = text

        return TaskCParsedOut(raw_address=raw, parsed=parsed)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_c/best.pt")
    ap.add_argument("--raw_address", type=str, required=True)
    ap.add_argument("--confidence", type=float, default=0.30)
    args = ap.parse_args()

    rt = TaskCRuntime(Path(args.ckpt))
    pred = rt.predict(args.raw_address, confidence_thresh=args.confidence)
    print(json.dumps({"raw_address": pred.raw_address, "parsed": pred.parsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
