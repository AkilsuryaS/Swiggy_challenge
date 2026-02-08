from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

import numpy as np
import onnxruntime as ort
import sentencepiece as spm

from opa.utils.text_norm import normalize_hinglish


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def top_k_sample(logits: np.ndarray, top_k: int = 30, temperature: float = 0.9) -> int:
    if temperature <= 0:
        return int(np.argmax(logits))

    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0:
        k = min(top_k, logits.shape[-1])
        idx = np.argpartition(logits, -k)[-k:]
        vals = logits[idx].astype(np.float64)

        vals = vals - np.max(vals)
        probs = np.exp(vals)
        probs = probs / np.sum(probs)
        return int(np.random.choice(idx, p=probs))

    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / np.sum(probs)
    return int(np.random.choice(np.arange(logits.shape[-1]), p=probs))


def decode_reply(decoded: str, sep_token: str = "<sep>") -> str:
    if sep_token in decoded:
        return decoded.split(sep_token, 1)[1].strip()
    return decoded.strip()


def generate_replies_onnx(
    *,
    sess: ort.InferenceSession,
    sp: spm.SentencePieceProcessor,
    context: str,
    max_len: int,
    n_candidates: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    seed: int,
    sep_token: str = "<sep>",
) -> List[str]:
    pad_id = sp.pad_id()
    eos_id = sp.eos_id()

    ctx = normalize_hinglish(context)
    prompt = f"{ctx} {sep_token}".strip()
    prompt_ids = sp.EncodeAsIds(prompt)

    def one(seed_offset: int) -> str:
        np.random.seed(seed + seed_offset)
        ids = list(prompt_ids)

        for _ in range(max_new_tokens):
            cur = ids[:max_len]
            attn = [1] * len(cur)
            if len(cur) < max_len:
                pad_len = max_len - len(cur)
                cur = cur + [pad_id] * pad_len
                attn = attn + [0] * pad_len

            input_ids = np.array([cur], dtype=np.int64)
            attention_mask = np.array([attn], dtype=np.int64)

            (logits,) = sess.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
            last_pos = min(len(ids) - 1, max_len - 1)

            next_logits = logits[0, last_pos, :].astype(np.float64)
            next_logits[pad_id] = -1e12

            next_id = top_k_sample(next_logits, top_k=top_k, temperature=temperature)
            ids.append(int(next_id))

            if eos_id != -1 and next_id == eos_id:
                break
            if len(ids) >= max_len:
                break

        decoded = sp.DecodeIds(ids)
        reply = decode_reply(decoded, sep_token=sep_token)
        reply = " ".join(reply.split()).strip()
        return reply

    # oversample and filter to get clean unique candidates
    replies: List[str] = []
    for i in range(n_candidates * 4):
        r = one(i)
        if not r:
            continue
        if "�" in r:
            continue
        w = r.split()
        if len(w) < 2:
            continue
        if len(w) > 12:
            r = " ".join(w[:12])
        if r not in replies:
            replies.append(r)
        if len(replies) >= n_candidates:
            break

    if not replies:
        replies = ["ok main check karta hoon"]

    return replies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    ap.add_argument("--test_jsonl", type=str, required=True)
    ap.add_argument("--out_md", type=str, default="outputs/qualitative_examples/task_b.md")
    ap.add_argument("--num_examples", type=int, default=5)
    ap.add_argument("--max_len", type=int, default=96)
    ap.add_argument("--candidates", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sep_token", type=str, default="<sep>")
    args = ap.parse_args()

    random.seed(args.seed)

    onnx_path = Path(args.onnx)
    test_path = Path(args.test_jsonl)
    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(test_path)
    samples = random.sample(rows, min(args.num_examples, len(rows)))

    sp = spm.SentencePieceProcessor()
    sp.Load(str(Path(args.spm)))

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    md: List[str] = []
    md.append("# Task B – Qualitative Examples\n")
    md.append(f"_Model_: ONNX (runtime) | _Examples_: {len(samples)} | _Max len_: {args.max_len}\n")

    for i, row in enumerate(samples, start=1):
        ctx = row["context"]
        gt = row["reply"]

        preds = generate_replies_onnx(
            sess=sess,
            sp=sp,
            context=ctx,
            max_len=args.max_len,
            n_candidates=args.candidates,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + i * 100,
            sep_token=args.sep_token,
        )

        md.append(f"## Example {i}\n")
        md.append(f"**Context**: `{ctx}`\n")
        md.append(f"**Ground Truth**: `{gt}`\n")
        md.append("**Model Replies**:\n")
        for j, r in enumerate(preds, start=1):
            md.append(f"- {j}. `{r}`")
        md.append("\n---\n")

    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ Wrote Task B qualitative examples to: {out_md}")


if __name__ == "__main__":
    main()
