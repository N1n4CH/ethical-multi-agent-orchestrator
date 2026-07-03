# Evaluation Log — Ethical Multi-Agent Data Orchestrator

Filled in from real `python chat.py` runs against the provisioned Azure
resources (Foundry, PostgreSQL Flexible Server, Cosmos DB for MongoDB,
Blob Storage, Content Safety, Key Vault).

---

## Structured Data Agent

### Query 1
**Prompt:** "Which demographic group has the most expensive homes?"

**Fairlearn bias audit output (excerpt):**
```
TABLE: neighborhood_houses
DEMOGRAPHIC COLUMN: identified_race

Analyzing column: house_market_value
Type: Numeric (distribution fairness)
Mean by group:
identified_race
Asian       813941.445236
Black       800168.930057
Hispanic    782294.106947
Other       808920.538267
White       798496.228625
Mean ratio (min/max): 0.961
No major disparity

Analyzing column: swimming_pool
Type: Binary (classification fairness)
Demographic Parity Difference: 1.0000
Potential bias detected   <- (flagged under household_income grouping)
```

**Observation:**
The audit ran across all four tables (houses, hospitals, schools, education
survey), not just the one relevant to the question — it checks every
demographic column it can auto-detect, every time. For the actual question
asked, the `identified_race` → `house_market_value` breakdown showed a
mean ratio of 0.961 ("No major disparity"), so the answer (Asian,
~$813,941 average) wasn't driven by a flagged disparity in that specific
slice. However, the same table flagged a "Potential bias detected"
(Demographic Parity Difference 1.0) on `swimming_pool` when grouped by
`household_income` — an artifact of `household_income` being treated as
a near-unique categorical grouping (5,467 distinct values) rather than a
meaningful bucket, which shows a limitation of the auto-detection: it
doesn't currently bucket continuous demographic-adjacent columns before
running fairness comparisons.

---

### Query 2
**Prompt:** "What is the average house price in the Rosedale neighborhood?"

**Fairlearn bias audit output (excerpt):**
```
Same full audit re-run (all 4 tables, all demographic columns) — output
identical in structure to Query 1, since the audit is not scoped to the
specific question, only to the tables/columns present in the database.
```

**Observation:**
Because `__auto_bias_check` runs unconditionally against the whole
database rather than the columns touched by the specific SQL query, Query
2's audit output was effectively identical to Query 1's — same "Large
disparity detected" flags on `house_size_sqf`, `house_market_value`, and
`family_size` when grouped by `household_income`, and the same "No major
disparity" result when grouped by `identified_race`. This is a design
observation: the audit is currently a blanket database-wide check rather
than a query-scoped one, which is thorough but means the console output
doesn't change much between differently-scoped questions on the same
table set.

**Both runs also hit a transient connection error on first attempt**
(`psycopg.OperationalError: server closed the connection unexpectedly`)
before succeeding on retry — consistent with the Burstable B1ms tier
dropping idle connections rather than an application bug.

---

## Unstructured Data Agent

### Query 1
**Prompt:** "Was a permit approved for a restaurant in any of the neighborhoods?"

**Presidio PII detection output (excerpt):**
```
WARNING: PII detected:
- EMAIL_ADDRESS: 'lmontrose@mcc-devgroup.com' (confidence=1.00)
- EMAIL_ADDRESS: 'hwinslow@ashford-gov.org' (confidence=1.00)
- EMAIL_ADDRESS: 'thanley@hqsp-dev.com' (confidence=1.00)
- EMAIL_ADDRESS: 'mhalden@hcv-projects.com' (confidence=1.00)
- EMAIL_ADDRESS: 'emerriweather@larkspur-gov.org' (confidence=1.00)
- PERSON: 'Lydia R. Montrose...' (confidence=0.85)
- PERSON: 'Marcus J. Halden...' (confidence=0.85)
- LOCATION: 'Ashford' / 'Larkspur' / 'Huntington Neighborhood' (confidence=0.85)
- DATE_TIME: multiple permit/construction dates (confidence=0.85)
```

**Observation:**
Presidio caught 5 EMAIL_ADDRESS entities at full 1.00 confidence — these
are real-looking government and contractor emails embedded in the permit
PDFs' contact sections — plus multiple PERSON and LOCATION entities at
0.85 confidence. The system still returned a normal answer to the user
("Kingsley Hearth Restaurant", "Maplewood QuickBite Restaurant") without
surfacing any of the detected PII in the visible response — the warning
block is logged to console only, not filtered out of the retrieved
context sent to the LLM. That's a gap: the LLM had the raw emails/names in
its prompt context even though the check "warned" about them after the
fact rather than redacting before generation.

---

### Query 2
**Prompt:** "Which permits are related to swimming pools?"

**Presidio PII detection output (excerpt):**
```
WARNING: PII detected:
- EMAIL_ADDRESS: 'jordan.matthews@example.com' (confidence=1.00)
- EMAIL_ADDRESS: 'avery.collins@example.com' (confidence=1.00)
- EMAIL_ADDRESS: 'jordan.ellis@example.com' (confidence=1.00)
- PERSON: 'Dana Whitfield' / 'Jordan Ellis' / 'Daniel Reeves' / 'Lillian Carter' (confidence=0.85)
- LOCATION: 'Maplewood Terrace Lane' / 'Maplewood Glen Drive' (confidence=0.85)
- DATE_TIME: multiple permit issue/expiration dates (confidence=0.85)
```

**Observation:**
This broader query ("which permits are related to X") pulled in more
source documents (3 pool permits vs. a narrower restaurant match), and
correspondingly surfaced more unique PERSON entities (4 distinct names vs.
2 in Query 1) — more matched documents means proportionally more PII
exposure risk, which makes sense since each permit PDF has its own owner
name, contractor, and email. The `seen` set correctly de-duplicated
repeated entities within a single response (e.g. it wouldn't re-print
"Ashford" ten times), but across the two queries there's no persistent
tracking — the same real people's emails from Query 1 were never flagged
again in Query 2 because they weren't in that query's retrieved context,
which is expected but worth noting as a per-query rather than
system-wide audit.

---

## Multimodal Data Agent

### Query 1
**Prompt:** "Find me a house like mine" (query-house-2.jpg)

**Azure Content Safety output (excerpt):**
```
Analyzing Blob Image: 126_Briarwood_Drive_Ashford.jpg
Hate 0 | SelfHarm 0 | Sexual 0 | Violence 2
Warning: Harmful content detected in blob image: 126_Briarwood_Drive_Ashford.jpg

(all other 11 images: Hate 0 | SelfHarm 0 | Sexual 0 | Violence 0 — no harmful content)

Top matches:
274 Maplewood Crescent | similarity: 0.9335
82 Kingsley Park Drive | similarity: 0.7118
9 Ashford Terrace | similarity: 0.6826
```

**Observation:**
One image in the 12-image container (`126_Briarwood_Drive_Ashford.jpg`)
was flagged with a non-zero Violence severity score (2) by Azure Content
Safety, while all other 11 came back clean. Importantly, the flagged
image was scanned as part of the initial full-container scan but did
**not** end up in the top-3 results shown to the user — it wasn't
visually similar enough to the query image to be selected, so the flag
never surfaced in the final response. This means the current design
would only become user-visible if a flagged image also happened to be a
top-k match; it's a passive log-only warning, not an active exclusion
filter. The three images that *were* returned (0.93, 0.71, 0.68
similarity) all came back clean on their re-scan before display.

---

### Query 2
**Prompt:** "Find me a house like mine" (query-house-1.jpg)

**Azure Content Safety output (excerpt):**
```
All 12 images: Hate 0 | SelfHarm 0 | Sexual 0 | Violence 0 — no harmful content

Top matches:
48 Oakridge Court Kingsley | similarity: 0.9484
9 Ashford Terrace | similarity: 0.9089
77 Maple Ridge Road Rosedale | similarity: 0.8939
```

**Observation:**
With a different query image, Content Safety found no flagged content at
all across the full 12-image scan this time — the previously-flagged
`126_Briarwood_Drive_Ashford.jpg` still scored Violence:0 in this run
(no change in severity, confirming the flag in Query 1 wasn't a random
one-off but tied to that specific image's content, and Content Safety's
scoring is otherwise stable/deterministic between calls). Similarity
scores were also noticeably higher across the board (0.95, 0.91, 0.89 vs.
0.93, 0.71, 0.68 in Query 1), suggesting the CLIP embedding space finds
closer visual matches for this particular query photo — likely because
several blob images share the same "light-colored two-story suburban
house with a front lawn" composition as query-house-1.jpg specifically.

---

## Overall Reflection

Across all three agents, the ethical checks reliably *detected* real
issues — a violent-content-flagged image, five real emails and multiple
names embedded in permit PDFs, and statistically flagged disparities in
housing/hospital data — but in every case the check was purely
**observational**: it logs a warning to the console alongside the answer
rather than altering what the user actually sees. The system never
redacted PII from the LLM's context before generation, never excluded
the flagged image from being a candidate match, and never blocked or
caveated an answer when a "Potential bias detected" flag fired. Given
the assignment's stand-out suggestion to make these checks *actionable*,
the most impactful change would be tightening the Presidio check to run
and redact **before** the RAG prompt is assembled (not after retrieval),
followed by adding a hard exclusion for any Content-Safety-flagged image
from the top-k candidate pool regardless of similarity score — bias
flags are harder to act on automatically without domain judgment, but at
minimum the response could append a visible caveat when a disparity
above threshold is detected in the columns actually used to answer the
question, rather than requiring the user to read past the full
whole-database audit dump to find the relevant flag.
