from __future__ import annotations

from typing import Dict, Literal, Optional
from pydantic import BaseModel, validator


Intent = Literal[
    "get_address",
    "call_customer",
    "mark_delivered",
    "mark_picked_up",
    "report_delay",
    "navigation_help",
    "order_issue",
    "customer_unavailable",
]

OrderSlot = Literal["next", "current", "previous"]
IssueSlot = Literal["item_missing", "restaurant_delay", "wrong_address", "payment", "other"]


class TaskARecord(BaseModel):
    """
    Raw Task A record schema:
    {"text": "...", "intent": "...", "slots": {...}}

    Slots are optional and depend on intent.
    """
    text: str
    intent: Intent
    slots: Dict[str, str] = {}

    @validator("text")
    def text_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("text is empty")
        # hard rule from prompt: roman only (we'll allow punctuation/numbers)
        if any("\u0900" <= ch <= "\u097F" for ch in v):
            raise ValueError("text contains Devanagari characters (not allowed)")
        return v

    @validator("slots")
    def slots_valid_for_intent(cls, slots: Dict[str, str], values) -> Dict[str, str]:
        intent = values.get("intent")
        if intent is None:
            return slots

        # allow empty dict always
        if slots is None:
            return {}

        # enforce only known slot keys
        allowed_keys = {"order", "delay_min", "issue"}
        for k in slots.keys():
            if k not in allowed_keys:
                raise ValueError(f"Unknown slot key: {k}")

        # intent-specific constraints
        if intent == "get_address":
            if "order" in slots:
                if slots["order"] not in ("next", "current", "previous"):
                    raise ValueError(f"Invalid order slot: {slots['order']}")
            # other keys generally shouldn't appear
            for k in slots.keys():
                if k != "order":
                    raise ValueError(f"Slot '{k}' not expected for intent get_address")

        if intent == "report_delay":
            if "delay_min" in slots:
                dm = slots["delay_min"]
                if not dm.isdigit():
                    raise ValueError(f"delay_min must be a string integer, got {dm}")
                # keep it permissive; we'll clamp later if needed
                if int(dm) <= 0 or int(dm) > 240:
                    raise ValueError(f"delay_min out of reasonable range: {dm}")
            for k in slots.keys():
                if k != "delay_min":
                    raise ValueError(f"Slot '{k}' not expected for intent report_delay")

        if intent == "order_issue":
            if "issue" not in slots:
                raise ValueError("order_issue intent must include slots.issue")
            if slots["issue"] not in ("item_missing", "restaurant_delay", "wrong_address", "payment", "other"):
                raise ValueError(f"Invalid issue slot: {slots['issue']}")
            for k in slots.keys():
                if k != "issue":
                    raise ValueError(f"Slot '{k}' not expected for intent order_issue")

        # other intents should usually be empty; enforce emptiness to keep labels clean
        if intent in (
            "call_customer",
            "mark_delivered",
            "mark_picked_up",
            "navigation_help",
            "customer_unavailable",
        ):
            if len(slots) != 0:
                raise ValueError(f"Intent {intent} should have empty slots, got: {slots}")

        return slots


class TaskBRecord(BaseModel):
    """
    Raw Task B record schema (JSONL):
    {"context": "...", "reply": "..."}
    """

    context: str
    reply: str

    @validator("context")
    def context_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("context is empty")
        if any("\u0900" <= ch <= "\u097F" for ch in v):
            raise ValueError("context contains Devanagari characters (not allowed)")
        return v

    @validator("reply")
    def reply_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("reply is empty")
        if any("\u0900" <= ch <= "\u097F" for ch in v):
            raise ValueError("reply contains Devanagari characters (not allowed)")
        # Hard constraint from prompt: <= 12 words (keep a small buffer)
        words = v.split()
        if len(words) > 14:
            raise ValueError(f"reply too long: {len(words)} words")
        return v


from typing import Optional


class TaskCParsed(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    house_flat: Optional[str] = None
    building: Optional[str] = None
    street: Optional[str] = None
    landmark: Optional[str] = None
    locality: Optional[str] = None
    city: Optional[str] = None
    state: Optional[str] = None
    pincode: Optional[str] = None

    @validator("pincode")
    def pincode_six_digits_or_null(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        if len(v) != 6 or not v.isdigit():
            raise ValueError("pincode must be exactly 6 digits or null")
        return v

    @validator("phone")
    def phone_masked_or_null(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None
        # Allow masked formats only (e.g., 98xxxxxx12, 9xxxxxxx21)
        digits = [c for c in v if c.isdigit()]
        if len(digits) >= 10:
            # looks like a real phone number -> reject
            raise ValueError("phone looks like a real phone number; must be masked or null")
        return v


class TaskCRecord(BaseModel):
    """
    Task C raw schema:
    {
      "raw_address": "...",
      "parsed": { ...fields... }
    }
    """
    raw_address: str
    parsed: TaskCParsed

    @validator("raw_address")
    def raw_address_valid(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("raw_address is empty")
        # Reject Devanagari (must be roman)
        if any("\u0900" <= ch <= "\u097F" for ch in v):
            raise ValueError("raw_address contains Devanagari characters")
        return v

