# Task A – Qualitative Examples

_Model_: INT8 ONNX | _Examples_: 5 | _Max length_: 64

## Example 1

**Input**: `sir restaurant late kar raha abhi complaint raise karo`

**Ground Truth**:

- Intent: `order_issue`

- Slots: `{"issue": "restaurant_delay"}`

**Model Output**:

- Intent: `order_issue`

- Slots: `{}`

✅ Intent correct


---

## Example 2

**Input**: `pls order pickup ho gaya jaldi picked up mark kar do`

**Ground Truth**:

- Intent: `mark_picked_up`

- Slots: `{}`

**Model Output**:

- Intent: `mark_picked_up`

- Slots: `{}`

✅ Intent correct


---

## Example 3

**Input**: `bro 15 mins late ho jaunga signal pe atak gaya`

**Ground Truth**:

- Intent: `report_delay`

- Slots: `{"delay_min": "15"}`

**Model Output**:

- Intent: `report_delay`

- Slots: `{}`

✅ Intent correct


---

## Example 4

**Input**: `aaj traffic heavy thoda delay ho gaya`

**Ground Truth**:

- Intent: `report_delay`

- Slots: `{}`

**Model Output**:

- Intent: `report_delay`

- Slots: `{}`

✅ Intent correct


---

## Example 5

**Input**: `bhai please lcation tak navigate karwa do plz`

**Ground Truth**:

- Intent: `navigation_help`

- Slots: `{}`

**Model Output**:

- Intent: `navigation_help`

- Slots: `{}`

✅ Intent correct


---
