Offline Partner Assistant 
=========================

This repository implements **Challenge 1** as an on-device, fully offline
language-model-based assistant for delivery partners.

The system consists of three tasks:
- **Task A**: Hinglish Command Parsing (intent + slots)
- **Task B**: Smart Reply Generation
- **Task C**: Indian Address Parsing

All models are:
- trained from scratch (no hosted APIs)
- optimized for on-device inference
- exported to ONNX and INT8 quantized
- runnable fully offline on CPU / Apple Silicon


Task A (Hinglish Command Parser)
================================

What this is
------------
This project implements Challenge 1 – Task A: an on-device, fully offline language model that parses Hinglish delivery partner commands into structured intent + slots.

Demo in One Line
---------------
Input:
bhai next order ka address batao

Output:
{"intent":"get_address","slots":{"order":"next"}}

Example
-------
Input:
```
order issue hai, item missing hai
```

Output:
```json
{"intent":"order_issue","slots":{"issue":"item_missing"}}
```

Supported Intents
-----------------
get_address
call_customer
mark_delivered
mark_picked_up
report_delay
navigation_help
order_issue
customer_unavailable

Slots
-----
order: next | current | previous
delay_min: string integer
issue: item_missing | restaurant_delay | wrong_address | payment | other

Tiny Transformer encoder, trained from scratch
Masked Language Modeling (MLM) auxiliary loss

Size / Speed
------------
~25 ms CPU latency (MacBook M1 Pro)
INT8 ONNX model ~3.2 MB
Fully offline

Data
----
~3,000 synthetic Hinglish commands

Generated via LLM with realistic:
slang
typos
short / long commands

Strict schema validation + deduplication

Setup
-----
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

Train
-----
python -m scripts.train_task_a --prefer_mps

Output:
models/task_a/best.pt

Export + Quantize
----------------
python -m scripts.export_onnx \
  --ckpt models/task_a/best.pt \
  --out_dir models/task_a/onnx

python -m scripts.benchmark_infer \
  --onnx models/task_a/onnx/task_a_intent_slot.onnx \
  --out_dir models/task_a/onnx

Run ONNX Inference (End-to-End)
-------------------------------
python -m scripts.run_task_a_onnx \
  --onnx models/task_a/onnx/task_a_intent_slot.int8.onnx \
  --spm models/tokenizer/task_a/spm.model \
  --label_maps data/processed/task_a/label_maps.json \
  --text "traffic zyada hai 10 min late hoga"

Output:
{"intent":"report_delay","slots":{"delay_min":"10"}}

Qualitative Results
-------------------
5 qualitative examples (ground truth vs model output) are auto-generated using INT8 ONNX inference:
outputs/qualitative_examples/task_a.md

---------


Task B (Smart Reply Generator)
===============================

What this is
------------
Task B generates short, natural Hinglish smart replies for a given delivery-partner context. Replies are constrained to be concise (<= 12 words) and fully offline.

Demo in One Line
---------------
Context:
customer gate band hai, kya karu

Output (example):
{"context":"customer gate band hai, kya karu","replies":["main call karta hoon","main wait karta hoon","aap gate khol sakte ho?"]}

Example
-------
Context:
```
traffic zyada hai, late hoga
```

Output (example):
```json
{"context":"traffic zyada hai, late hoga","replies":["10 min late ho jaunga","thoda delay hai","main pahunch raha hoon"]}
```

Data
----
Synthetic Hinglish context-reply pairs with strict schema validation.

Tokenizer
---------
python -m scripts.build_tokenizer_b \
  --task_b_clean_jsonl data/interim/task_b/clean_v2.jsonl \
  --out_dir models/tokenizer/task_b

Preprocess
----------
python -m scripts.preprocess \
  --task task_b \
  --clean_jsonl data/interim/task_b/clean_v2.jsonl \
  --out_dir data/processed/task_b

Train
-----
python -m scripts.train_task_b --prefer_mps

Output:
models/task_b/best.pt

Export + Quantize
----------------
python -m scripts.export_task_b_onnx \
  --ckpt models/task_b/best.pt \
  --out_dir models/task_b/onnx

Run ONNX Inference (End-to-End)
-------------------------------
python -m scripts.run_task_b_onnx \
  --onnx models/task_b/onnx/task_b_lm.int8.onnx \
  --spm models/tokenizer/task_b/spm.model \
  --context "customer gate band hai, kya karu"

Output:
{"context":"customer gate band hai, kya karu","replies":["main call karta hoon","main wait karta hoon","aap gate khol sakte ho?"]}

Qualitative Results
-------------------
python -m scripts.evaluate_task_b \
  --onnx models/task_b/onnx/task_b_lm.onnx \
  --spm models/tokenizer/task_b/spm.model \
  --test_jsonl data/processed/task_b/test.jsonl

Outputs:
outputs/qualitative_examples/task_b.md


Task C (Address Parser)
=======================

What this is
------------
Task C parses a raw delivery address into structured fields (name/phone/address parts) using a lightweight encoder with span prediction and MLM auxiliary loss.

Demo in One Line
---------------
Raw:
flat 2B, MG road, Indore 452001

Output (example):
{"raw_address":"flat 2B, MG road, Indore 452001","parsed":{"name":null,"phone":null,"house_flat":"2B","building":null,"street":"MG road","landmark":null,"locality":null,"city":"Indore","state":null,"pincode":"452001"}}

Example
-------
Raw:
```
2nd floor, A wing, Green Tower, Andheri West Mumbai 400058
```

Output (example):
```json
{"raw_address":"2nd floor, A wing, Green Tower, Andheri West Mumbai 400058","parsed":{"name":null,"phone":null,"house_flat":"2nd floor","building":"Green Tower","street":null,"landmark":null,"locality":"Andheri West","city":"Mumbai","state":null,"pincode":"400058"}}
```

Data
----
Synthetic Hinglish/roman address data with strict schema validation and masked phone enforcement.

Preprocess
----------
python -m scripts.preprocess \
  --task task_c \
  --clean_jsonl data/interim/task_c/clean_v3.jsonl \
  --out_dir data/processed/task_c

Train
-----
python -m scripts.train_task_c --prefer_mps

Output:
models/task_c/best.pt

Export ONNX
-----------
python -m scripts.export_task_c_onnx \
  --ckpt models/task_c/best.pt \
  --out_dir models/task_c/onnx

Run Inference (PyTorch)
-----------------------
python -m scripts.run_task_c \
  --ckpt models/task_c/best.pt \
  --raw_address "flat 2B, MG road, Indore 452001"

Run Inference (ONNX)
-------------------
python -m scripts.run_task_c_onnx \
  --onnx models/task_c/onnx/task_c_spans.onnx \
  --spm models/tokenizer/task_a/spm.model \
  --raw_address "flat 2B, MG road, Indore 452001"

Qualitative Results
-------------------
python -m scripts.evaluate_task_c \
  --onnx models/task_c/onnx/task_c_spans.onnx \
  --spm models/tokenizer/task_a/spm.model \
  --test_jsonl data/processed/task_c/test.jsonl \
  --out_md outputs/qualitative_examples/task_c.md \
  --num_examples 5

Outputs:
outputs/qualitative_examples/task_c.md


Performance Summary (CPU, batch=1)
----------------------------------
| Task | Model Size (INT8) | Latency |
|------|------------------|---------|
| A    | ~3.2 MB          | ~25 ms  |
| B    | ~3.1 MB          | ~30 ms  |
| C    | ~4.0 MB          | ~35 ms  |
