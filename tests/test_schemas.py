from __future__ import annotations

import pytest

from opa.data.schemas import TaskARecord, TaskBRecord, TaskCParsed, TaskCRecord


def test_task_a_valid_record():
    rec = TaskARecord(text="bhai address bhejo", intent="get_address", slots={"order": "next"})
    assert rec.intent == "get_address"
    assert rec.slots["order"] == "next"


def test_task_a_invalid_slot_key():
    with pytest.raises(ValueError):
        TaskARecord(text="ok", intent="get_address", slots={"foo": "bar"})


def test_task_a_invalid_delay():
    with pytest.raises(ValueError):
        TaskARecord(text="late 10", intent="report_delay", slots={"delay_min": "0"})


def test_task_b_valid_record():
    rec = TaskBRecord(context="gate band hai", reply="ok main check karta hoon")
    assert rec.context
    assert rec.reply


def test_task_b_rejects_devanagari():
    with pytest.raises(ValueError):
        TaskBRecord(context="नमस्ते", reply="ok")


def test_task_b_rejects_long_reply():
    long_reply = " ".join(["ok"] * 20)
    with pytest.raises(ValueError):
        TaskBRecord(context="test", reply=long_reply)


def test_task_c_parsed_pincode_validation():
    with pytest.raises(ValueError):
        TaskCParsed(pincode="123")


def test_task_c_parsed_phone_validation():
    with pytest.raises(ValueError):
        TaskCParsed(phone="9876543210")


def test_task_c_record_rejects_devanagari_raw():
    with pytest.raises(ValueError):
        TaskCRecord(raw_address="दिल्ली", parsed=TaskCParsed())
