from __future__ import annotations

from pathlib import Path

from opa.tokenization.tokenizer import SentencePieceTokenizer


def _model_path() -> Path:
    root = Path(__file__).resolve().parents[1]
    return root / "models" / "tokenizer" / "task_a" / "spm.model"


def test_encode_decode_roundtrip():
    tok = SentencePieceTokenizer(_model_path())
    text = "hello bhai kaise ho"
    ids = tok.encode(text, add_bos=False, add_eos=False)
    out = tok.decode(ids)
    assert isinstance(out, str)
    assert out.replace(" ", "") != ""


def test_encode_adds_eos_when_enabled():
    tok = SentencePieceTokenizer(_model_path())
    text = "test"
    ids = tok.encode(text, add_bos=False, add_eos=True)
    if tok.eos_id != -1:
        assert ids[-1] == tok.eos_id


def test_encode_batch_pad_shapes_and_mask():
    tok = SentencePieceTokenizer(_model_path())
    texts = ["hello", "hello world"]
    max_len = 8
    ids, mask = tok.encode_batch_pad(texts, max_length=max_len, add_bos=False, add_eos=True)
    assert len(ids) == len(texts)
    assert len(mask) == len(texts)
    assert all(len(x) == max_len for x in ids)
    assert all(len(x) == max_len for x in mask)
    # attention mask should be 1 for non-pad positions and 0 for pad
    for row_ids, row_mask in zip(ids, mask):
        for i, mid in enumerate(row_mask):
            if mid == 0:
                assert row_ids[i] == tok.pad_id
