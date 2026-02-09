# Task C – Qualitative Examples

_Model_: ONNX | _Examples_: 5 | _Max len_: 128

## Example 1

**Raw**: `imran h-983, 3rd flr sunshine ph 5 apartments nr hospital mylapore chennai tamil nadu 600017`

**Ground Truth Parsed**:

```json
{
  "name": "imran",
  "phone": null,
  "house_flat": "h-983, 3rd flr",
  "building": "sunshine ph 5 apartments",
  "street": null,
  "landmark": "nr hospital",
  "locality": "mylapore",
  "city": "chennai",
  "state": "tamil nadu",
  "pincode": "600017"
}
```

**Model Parsed**:

```json
{
  "name": "imran",
  "phone": null,
  "house_flat": "h-983, 3rd flr",
  "building": "sunshine ph 5 apartments",
  "street": null,
  "landmark": "nr hospital",
  "locality": "mylapore chennai",
  "city": "chennai tami",
  "state": "tamil nadu",
  "pincode": "600017"
}
```

---

## Example 2

**Raw**: `h no 201 hinjewadi phase 2 pune maharashtra 411045`

**Ground Truth Parsed**:

```json
{
  "name": null,
  "phone": null,
  "house_flat": "h no 201",
  "building": null,
  "street": null,
  "landmark": null,
  "locality": "hinjewadi phase 2",
  "city": "pune",
  "state": "maharashtra",
  "pincode": "411045"
}
```

**Model Parsed**:

```json
{
  "name": null,
  "phone": null,
  "house_flat": "h no 201",
  "building": null,
  "street": null,
  "landmark": "2",
  "locality": "2",
  "city": "pune",
  "state": "maharashtra",
  "pincode": "411045"
}
```

---

## Example 3

**Raw**: `a-455 opp metro stn dadar mumbai maharashtra 400014`

**Ground Truth Parsed**:

```json
{
  "name": null,
  "phone": null,
  "house_flat": "a-455",
  "building": null,
  "street": "church cross",
  "landmark": "opp metro stn",
  "locality": "dadar",
  "city": "mumbai",
  "state": "maharashtra",
  "pincode": "400014"
}
```

**Model Parsed**:

```json
{
  "name": "a-45",
  "phone": null,
  "house_flat": "a-455",
  "building": "stn da",
  "street": "metro st",
  "landmark": "opp metro stn",
  "locality": "dadar",
  "city": "mumbai",
  "state": "maharashtra",
  "pincode": "400014"
}
```

---

## Example 4

**Raw**: `ph 91xxxxxx09, e-1672, outer ring nagar, dadar, mumbai, maharashtra, 400062`

**Ground Truth Parsed**:

```json
{
  "name": null,
  "phone": "91xxxxxx09",
  "house_flat": "e-1672",
  "building": null,
  "street": "outer ring nagar",
  "landmark": null,
  "locality": "dadar",
  "city": "mumbai",
  "state": "maharashtra",
  "pincode": "400062"
}
```

**Model Parsed**:

```json
{
  "name": null,
  "phone": "91xxxxxx09",
  "house_flat": "e-1672",
  "building": "outer ring",
  "street": "outer ring nagar",
  "landmark": "e-1672, outer ring nagar",
  "locality": "dadar",
  "city": "mumbai",
  "state": "maharashtra",
  "pincode": "400062"
}
```

---

## Example 5

**Raw**: `gulmohar heights market street beside market kukatpally 500072`

**Ground Truth Parsed**:

```json
{
  "name": null,
  "phone": null,
  "house_flat": null,
  "building": "gulmohar heights",
  "street": "market street",
  "landmark": "beside market",
  "locality": "kukatpally",
  "city": null,
  "state": null,
  "pincode": "500072"
}
```

**Model Parsed**:

```json
{
  "name": null,
  "phone": null,
  "house_flat": "gulmohar",
  "building": "gulmohar heights",
  "street": null,
  "landmark": null,
  "locality": null,
  "city": "kukatpal",
  "state": null,
  "pincode": "500072"
}
```

---
