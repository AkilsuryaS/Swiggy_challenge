from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish


def top_k_sample(logits: np.ndarray, top_k: int = 30, temperature: float = 0.9) -> int:
    # logits: [V]
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        idx = np.argpartition(logits, -k)[-k:]
        vals = logits[idx]
        # softmax
        vals = vals - np.max(vals)
        probs = np.exp(vals)
        probs = probs / np.sum(probs)
        choice = np.random.choice(idx, p=probs)
        return int(choice)

    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(logits.shape[-1]), p=probs))


def decode_reply(decoded: str, sep_token: str = "<sep>") -> str:
    if sep_token in decoded:
        return decoded.split(sep_token, 1)[1].strip()
    return decoded.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True, help="models/task_b/onnx/task_b_lm.onnx or int8 variant")
    ap.add_argument("--spm", type=str, required=True, help="models/tokenizer/task_b/spm.model")
    ap.add_argument("--context", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep_token", type=str, default="<sep>")
    args = ap.parse_args()

    np.random.seed(args.seed)

    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(args.spm)))

    pad_id = sp.pad_id()
    eos_id = sp.eos_id()

    sess = ort.InferenceSession(str(Path(args.onnx)), providers=["CPUExecutionProvider"])

    ctx = normalize_hinglish(args.context)
    prompt = f"{ctx} {args.sep_token}".strip()

    # prompt ids (no eos)
    prompt_ids = sp.EncodeAsIds(prompt)

    def generate_one(offset_seed: int) -> str:
        np.random.seed(args.seed + offset_seed)
        ids = list(prompt_ids)

        for _ in range(args.max_new_tokens):
            cur = ids[: args.max_len]
            attn = [1] * len(cur)
            if len(cur) < args.max_len:
                pad_len = args.max_len - len(cur)
                cur = cur + [pad_id] * pad_len
                attn = attn + [0] * pad_len

            input_ids = np.array([cur], dtype=np.int64)
            attention_mask = np.array([attn], dtype=np.int64)

            (logits,) = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            # logits: [1, T, V]
            last_pos = min(len(ids) - 1, args.max_len - 1)
            next_logits = logits[0, last_pos, :].astype(np.float64)

            # ban pad token
            next_logits[pad_id] = -1e12

            next_id = top_k_sample(next_logits, top_k=args.top_k, temperature=args.temperature)
            ids.append(int(next_id))

            if eos_id != -1 and next_id == eos_id:
                break
            if len(ids) >= args.max_len:
                break

        decoded = sp.DecodeIds(ids)
        reply = decode_reply(decoded, sep_token=args.sep_token)
        reply = " ".join(reply.split()).strip()
        return reply

    replies: List[str] = []
    for i in range(args.candidates * 3):  # oversample then dedupe
        r = generate_one(i)
        if not r:
            continue
        # keep short smart replies
        w = r.split()
        if len(w) < 2:
            continue
        if len(w) > 12:
            r = " ".join(w[:12])
        if r not in replies:
            replies.append(r)
        if len(replies) >= args.candidates:
            break

    if not replies:
        replies = ["ok main check karta hoon"]

    print(json.dumps({"context": args.context, "replies": replies}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
