You are generating training data for Task C: Indian Address Parser
for an offline delivery partner assistant.

GOAL:
Generate EXACTLY 3000 unique examples of messy Indian addresses (roman script + common abbreviations/typos)
paired with structured parsed fields.

IMPORTANT ITERATION RULE:
- Generate data in internal batches.
- Keep a running internal count of how many examples you have produced.
- If you have not yet reached 3000 examples, continue generating more.
- Do NOT stop early.
- Stop ONLY when you have produced exactly 3000 valid JSON objects.

--------------------------------
OUTPUT FORMAT (STRICT JSONL)
--------------------------------
- Output ONLY JSONL
- One JSON object per line
- No commas between lines
- No markdown
- No explanations
- No numbering
- No blank lines

Schema (exact keys, no extras):
{
  "raw_address": "string",
  "parsed": {
    "name": "string | null",
    "phone": "string | null",
    "house_flat": "string | null",
    "building": "string | null",
    "street": "string | null",
    "landmark": "string | null",
    "locality": "string | null",
    "city": "string | null",
    "state": "string | null",
    "pincode": "string | null"
  }
}

--------------------------------
FEW-SHOT EXAMPLES (STYLE REFERENCE)
--------------------------------
{"raw_address":"Ravi, 2nd flr, H no 14, near Hanuman mandir, Indiranagar, Bengaluru 560038","parsed":{"name":"Ravi","phone":null,"house_flat":"H no 14, 2nd flr","building":null,"street":null,"landmark":"near Hanuman mandir","locality":"Indiranagar","city":"Bengaluru","state":"Karnataka","pincode":"560038"}}
{"raw_address":"Flat 402, Sunshine Apts, MG Rd, opp Metro Stn, Pune - 411001","parsed":{"name":null,"phone":null,"house_flat":"Flat 402","building":"Sunshine Apts","street":"MG Rd","landmark":"opp Metro Stn","locality":null,"city":"Pune","state":"Maharashtra","pincode":"411001"}}
{"raw_address":"Sana (ph: 98xxxxxx12), B-7, Gulmohar Society, behind Big Bazaar, Andheri West, Mumbai 400058","parsed":{"name":"Sana","phone":null,"house_flat":"B-7","building":"Gulmohar Society","street":null,"landmark":"behind Big Bazaar","locality":"Andheri West","city":"Mumbai","state":"Maharashtra","pincode":"400058"}}

NOTE: NEVER put real phone numbers. If phone appears, use masked like "98xxxxxx12" or null.

--------------------------------
ADDRESS REALISM REQUIREMENTS
--------------------------------
Generate addresses from diverse Indian cities and states (mix metros + tier-2):
- Bengaluru, Mumbai, Delhi, Hyderabad, Chennai, Pune, Kolkata, Ahmedabad
- Jaipur, Lucknow, Indore, Bhopal, Surat, Kochi, Coimbatore, Nagpur, Patna, Bhubaneswar etc.

Include common patterns:
- House/Flat formats: "H.No 12", "House #12", "Flat 3B", "B-1203", "D-404", "2nd floor"
- Buildings: "Apts", "Society", "Residency", "Heights", "Tower", "Phase"
- Streets: "Rd", "Road", "Street", "St", "Main Rd", "Cross", "Lane", "Nagar", "Marg"
- Landmarks: "near", "opp", "beside", "behind", "next to"
- Localities: "Andheri West", "HSR Layout", "Koramangala", "Kothrud", etc.
- Pincode: 6 digits or null (not always present)

Introduce messiness:
- extra commas / missing commas
- mild typos: "apprtment", "socitey", "opp.", "nr"
- mixed casing
- variable ordering of parts
- occasional inclusion of a person's name at start
- occasional mention of "gate", "block", "wing"

--------------------------------
FIELD RULES (IMPORTANT)
--------------------------------
- All text must be Roman script (no Devanagari)
- phone must be null OR masked like "98xxxxxx12" (never real)
- pincode must be exactly 6 digits if present, else null
- state should be a real Indian state name if present (e.g., "Karnataka", "Maharashtra")
- city should be plausible for the locality/state
- parsed fields should be conservative:
  - If unsure, set field to null
  - Do not hallucinate pincode if not in raw_address
- raw_address should contain the information used to fill parsed fields (except that phone is masked)

--------------------------------
UNIQUENESS
--------------------------------
- Avoid duplicates
- Vary cities, localities, landmarks, formats
- Ensure each raw_address looks different

--------------------------------
FINAL VALIDATION BEFORE OUTPUT
--------------------------------
Before outputting:
- Ensure total lines = EXACTLY 3000
- Ensure each line is valid JSON
- Ensure keys match schema exactly (no missing keys)
- Ensure Roman script only
- Ensure no real phone numbers

Begin generation now and continue until all 3000 examples are completed.
