You are generating training data for an offline delivery partner assistant used in India.

GOAL:
Generate EXACTLY 3000 unique Hinglish command examples.

IMPORTANT ITERATION RULE:
- Generate data in internal batches.
- Keep a running internal count of how many examples you have produced.
- If you have not yet reached 3000 examples, continue generating more.
- Do NOT stop early.
- Stop ONLY when you have produced exactly 3000 valid JSON objects.

--------------------------------
FEW-SHOT EXAMPLES (STYLE REFERENCE)
--------------------------------

{"text":"bhai next order ka address dikha de","intent":"get_address","slots":{"order":"next"}}
{"text":"customer ko call laga de please","intent":"call_customer","slots":{}}
{"text":"traffic zyada hai 15 min late hoga","intent":"report_delay","slots":{"delay_min":"15"}}
{"text":"order delivered ho gaya mark kar do","intent":"mark_delivered","slots":{}}
{"text":"item missing lag raha hai, issue daal do","intent":"order_issue","slots":{"issue":"item_missing"}}
{"text":"maps open karo aur location pe le chalo","intent":"navigation_help","slots":{}}
{"text":"customer phone nahi utha raha","intent":"customer_unavailable","slots":{}}

--------------------------------
INTENTS (DISTRIBUTE EVENLY)
--------------------------------
- get_address
- call_customer
- mark_delivered
- mark_picked_up
- report_delay
- navigation_help
- order_issue
- customer_unavailable

--------------------------------
OUTPUT FORMAT (STRICT)
--------------------------------
- Output ONLY JSONL
- One JSON object per line
- No commas between lines
- No markdown
- No explanations
- No numbering
- No blank lines

Schema:
{"text": "...", "intent": "...", "slots": {...}}

--------------------------------
SLOT RULES
--------------------------------

get_address:
- If the order is mentioned, include:
  slots.order = "next" | "current" | "previous"
- If not mentioned, use {}

report_delay:
- If a delay time is mentioned, include:
  slots.delay_min = string integer ("5","10","15","20","30")
- If no time is mentioned, use {}

order_issue:
- Include:
  slots.issue = "item_missing" | "restaurant_delay" | "wrong_address" | "payment" | "other"

All other intents:
- slots should usually be {}

--------------------------------
LANGUAGE & STYLE RULES
--------------------------------
- Hinglish only (Hindi + English mixed naturally)
- Roman script ONLY (no Devanagari)
- Sound like real Indian delivery partners
- Mix tones:
  - casual
  - rushed
  - polite
  - frustrated
- Sentence length: 2–15 words
- Include realistic:
  - typos
  - abbreviations
  - informal grammar
- Avoid repeating sentence templates
- Avoid near-duplicates with minor word changes

--------------------------------
SAFETY & QUALITY CONSTRAINTS
--------------------------------
- No real phone numbers
- No real names
- No real addresses
- No emojis
- No additional fields
- No metadata
- No comments

--------------------------------
FINAL VALIDATION BEFORE OUTPUT
--------------------------------
Before outputting the final result:
- Ensure total number of lines is EXACTLY 3000
- Ensure every line is valid JSON
- Ensure intent values are only from the allowed list
- Ensure slot values follow the rules above

Begin generation now and continue until all 3000 examples are completed.
