You are generating training data for Task B: Smart Reply Generation
for an offline delivery partner assistant in India.

GOAL:
Generate EXACTLY 3000 unique examples of short smart replies in Hinglish.

Each example consists of:
- a short conversation context (what customer or system said)
- a concise reply that a delivery partner assistant would suggest

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

Schema:
{
  "context": "...",
  "reply": "..."
}

--------------------------------
FEW-SHOT EXAMPLES (STYLE REFERENCE)
--------------------------------

{"context":"customer bol raha hai gate band hai","reply":"main guard se baat karta hoon"}
{"context":"order thoda late ho raha hai","reply":"5 minute mein pahunch raha hoon"}
{"context":"customer call nahi utha raha","reply":"main thodi der mein dobara try karta hoon"}
{"context":"restaurant abhi food ready kar raha hai","reply":"jaise hi ready hoga main nikal jaunga"}
{"context":"location samajh nahi aa rahi","reply":"main map check karke aa raha hoon"}

--------------------------------
LANGUAGE & STYLE RULES
--------------------------------
- Hinglish only (Hindi + English mixed naturally)
- Roman script ONLY (no Devanagari)
- Replies must be SHORT:
  - 2 to 12 words max
- Tone should sound like a real delivery partner:
  - polite
  - reassuring
  - casual
  - sometimes apologetic
- Contexts may be:
  - customer messages
  - system notifications
  - delivery situations
- Replies must feel appropriate and natural for the context

--------------------------------
DIVERSITY REQUIREMENTS
--------------------------------
- Avoid repeating reply templates
- Vary:
  - sentence structure
  - verb usage
  - politeness level
- Include:
  - confirmations
  - apologies
  - status updates
  - reassurance
- Avoid generic chatbot phrasing

--------------------------------
SAFETY & CONSTRAINTS
--------------------------------
- No real phone numbers
- No real names
- No real addresses
- No emojis
- No abusive or unsafe language
- No extra fields beyond "context" and "reply"

--------------------------------
FINAL VALIDATION BEFORE OUTPUT
--------------------------------
Before outputting the final result:
- Ensure total number of lines is EXACTLY 3000
- Ensure every line is valid JSON
- Ensure every reply is <= 12 words
- Ensure Hinglish + Roman script only

Begin generation now and continue until all 3000 examples are completed.
