from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import torch

from opa.models.task_b.model import TinyCausalTransformerLM
from opa.models.task_b.decoding import DecodeConfig, top_k_sample, strip_after_sep
from opa.tokenization.tokenizer import SentencePieceTokenizer
from opa.utils.text_norm import normalize_hinglish


@dataclass
class TaskBGeneration:
    replies: List[str]


class TaskBRuntime:
    def __init__(self, ckpt_path: Path, device: Optional[str] = None, sep_token: str = "<sep>"):
        ckpt_path = Path(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.sep_token = sep_token
        self.sp_model_path = Path(ckpt["sp_model_path"])
        self.tokenizer = SentencePieceTokenizer(self.sp_model_path)
        self.max_len = int(ckpt["max_len"])
        cfg = ckpt["model_config"]

        self.model = TinyCausalTransformerLM(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
            d_model=int(cfg["d_model"]),
            n_heads=int(cfg["n_heads"]),
            n_layers=int(cfg["n_layers"]),
            d_ff=int(cfg["d_ff"]),
            dropout=0.0,
            max_len=int(cfg["max_len"]),
            tie_lm_head=bool(cfg.get("tie_lm_head", True)),
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

        # token id for "<sep>" if present; we ban generating it after prompt
        sep_ids = self.tokenizer.sp.EncodeAsIds(self.sep_token)
        self.sep_id = sep_ids[0] if len(sep_ids) == 1 else None

    @torch.no_grad()
    def generate(
        self,
        context: str,
        *,
        n_candidates: int = 3,
        decode_cfg: Optional[DecodeConfig] = None,
        seed: int = 42,
        min_reply_words: int = 2,
        max_reply_words: int = 12,
        repetition_penalty: float = 1.2,
        no_repeat_ngram: int = 3,
    ) -> TaskBGeneration:
        decode_cfg = decode_cfg or DecodeConfig()
        ctx = normalize_hinglish(context)
        prompt = f"{ctx} {self.sep_token}".strip()

        prompt_ids = self.tokenizer.encode(prompt, add_bos=False, add_eos=False, max_length=self.max_len)

        replies: List[str] = []
        for i in range(n_candidates):
            torch.manual_seed(seed + i)

            ids = list(prompt_ids)
            generated: List[int] = []
            seen_ngrams: Set[tuple] = set()

            for _ in range(decode_cfg.max_new_tokens):
                cur = ids[: self.max_len]
                attn = [1] * len(cur)
                if len(cur) < self.max_len:
                    pad_len = self.max_len - len(cur)
                    cur = cur + [self.tokenizer.pad_id] * pad_len
                    attn = attn + [0] * pad_len

                input_ids = torch.tensor([cur], dtype=torch.long, device=self.device)
                attention_mask = torch.tensor([attn], dtype=torch.long, device=self.device)

                out = self.model(input_ids=input_ids, attention_mask=attention_mask)
                last_pos = min(len(ids) - 1, self.max_len - 1)
                logits = out.logits[0, last_pos, :].clone()

                # ban pad and (after prompt) ban <sep> spam
                logits[self.tokenizer.pad_id] = float("-inf")
                if self.sep_id is not None:
                    logits[self.sep_id] = float("-inf")

                # repetition penalty
                if repetition_penalty and repetition_penalty > 1.0:
                    for tid in set(generated):
                        logits[tid] = logits[tid] / repetition_penalty

                # no-repeat ngram (simple)
                if no_repeat_ngram and no_repeat_ngram >= 2 and len(generated) >= no_repeat_ngram - 1:
                    prefix = tuple(generated[-(no_repeat_ngram - 1):])
                    # if any ngram already seen with this prefix, ban its next token
                    banned = [ng[-1] for ng in seen_ngrams if ng[:-1] == prefix]
                    for b in banned:
                        logits[b] = float("-inf")

                next_id = top_k_sample(logits, top_k=decode_cfg.top_k, temperature=decode_cfg.temperature)
                ids.append(next_id)
                generated.append(next_id)

                # update seen ngrams
                if no_repeat_ngram and no_repeat_ngram >= 2 and len(generated) >= no_repeat_ngram:
                    ng = tuple(generated[-no_repeat_ngram:])
                    seen_ngrams.add(ng)

                if decode_cfg.stop_on_eos and next_id == self.tokenizer.eos_id:
                    break
                if len(ids) >= self.max_len:
                    break

            decoded = self.tokenizer.decode(ids)
            reply = strip_after_sep(decoded, sep_token=self.sep_token)
            reply = " ".join(reply.split()).strip()

            w = reply.split()
            if len(w) < min_reply_words:
                continue
            if len(w) > max_reply_words:
                reply = " ".join(w[:max_reply_words])

            if reply and reply not in replies:
                replies.append(reply)

        if not replies:
            replies = ["ok main check karta hoon"]

        return TaskBGeneration(replies=replies)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="models/task_b/best.pt")
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    args = ap.parse_args()

    rt = TaskBRuntime(Path(args.ckpt))
    out = rt.generate(
        args.context,
        n_candidates=args.candidates,
        decode_cfg=DecodeConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            stop_on_eos=True,
        ),
    )
    print(json.dumps({"context": args.context, "replies": out.replies}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
