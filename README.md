Offline Partner Assistant – Task A (Hinglish Command Parser)
============================================================

What this is
------------
This project implements Challenge 1 – Task A: an on-device, fully offline language model that parses Hinglish delivery partner commands into structured intent + slots.

Demo in One Line
---------------
Input:
bhai next order ka address batao

Output:
{"intent":"get_address","slots":{"order":"next"}}

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

✅ Challenge 1 – Task A completed
