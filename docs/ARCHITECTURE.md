# Architecture — Mongolian Multi-Agent Health Chatbot (MVP v0.1)

Priority order for every design decision: **SAFETY > ACCURACY > RELIABILITY > COST > SPEED > COMPLEXITY**.

## 1. Turn pipeline (deterministic orchestrator — `backend/services/orchestrator.py`)

```
user text
  -> Input Processor            redact PII, Latin->Cyrillic, fold ө/ү, lexicon concepts, regex facts
  -> Agent 1 Triage, pass 1     YAML rules on the current message only
        EMERGENCY -> pre-approved template (103) -> Safety rules -> user        [0 LLM calls, ~5 ms]
  -> Agent 2 Symptom            intent, session merge, answers to pending questions, optional LLM extraction
  -> Agent 1 Triage, pass 2     session-merged concepts + facts; LLM risk may RAISE, never LOWER
        EMERGENCY -> template
  -> greeting / thanks / out-of-scope -> canned reply
  -> follow-up?                 symptom_check AND risk <= MODERATE AND missing slots AND rounds < 2
                                questions come from data/protocols (never invented by an LLM)
  -> Agent 3 RAG                hybrid search; insufficient -> safe fallback + knowledge_gaps row
  -> Agent 4 Response           template (offline) or LLM (grounded, cited)
  -> Agent 5 Safety             rules always; LLM judge for LLM drafts
        REVISE -> regenerate with issue list (max 2) -> still unsafe -> template answer -> fallback
  -> persist redacted text + full trace (versions, timings, token usage)
```

Agents never call each other. They exchange typed Pydantic objects (`backend/models/schemas.py`):
`NormalizedInput -> TriageResult -> SymptomAnalysis -> RetrievalResult -> DraftResponse -> SafetyVerdict`.

## 2. Where rules beat LLMs (and where they don't)

| Deterministic rules | LLM |
|---|---|
| Red-flag detection (recall-critical, auditable, works when the API is down) | Messy/informal language understanding beyond the lexicon |
| Emergency message text (pre-approved) | Fluent, friendly Mongolian wording |
| Follow-up questions (reviewed protocols) | Extraction of unusual phrasings |
| Citation/URL verification (set lookup) | Subtle safety issues (implied diagnosis, tone) |
| Dose / diagnosis / "stop your medicine" patterns | — |
| PII redaction, status filtering, routing | — |

## 3. Fail-closed policy
* Template answers are assembled only from knowledge-base text → must pass the rule checks.
* LLM answers must pass the rule checks **and** the LLM judge. If the judge is unavailable, the
  LLM answer is discarded and the template answer is used.
* Any LLM error (network, auth, refusal, invalid JSON) → deterministic path. The chatbot never
  depends on the LLM being up.
* Safety-critical concepts (`suicidal`) ignore negation; false alarms are cheaper than misses.
* Fuzzy (typo-tolerant) matching requires the same first letter and is skipped for words that are
  already in the lexicon (prevents "агаар" (air) → "угаар" (CO), "халуун" (hot) → "халуурах" (fever)).

## 4. Knowledge base & data versioning
* Markdown + YAML frontmatter, one file per topic, 4 fixed sections → one chunk per section.
* Lifecycle `DRAFT → REVIEW → VALIDATED → ACTIVE → DEPRECATED`; VALIDATED/ACTIVE require
  `reviewer` + `last_reviewed`. Production loads only ACTIVE.
* `data/metadata/kb_registry.json` is regenerated on every index build; `CHANGELOG.md` is manual.
* Medical logic (rules, protocols, lexicon, templates) is data, reviewed and versioned like documents.

## 5. Retrieval
* Score = `(0.6 × concept overlap + 0.4 × RRF(vector, BM25)) × tier weight` (+0.1 for medication docs
  on medication questions; −0.15 for docs about child/pregnancy/elderly when the user did not mention it).
* Sufficient evidence = Tier ≤ 2 AND (≥1 concept match OR cosine ≥ `COSINE_THRESHOLD`=0.5,
  calibrated on unknown-topic tests).
* **Index versioning (blue/green):** index name = embedding model + KB content hash. A new index is
  built next to the old one, retrieval tests run, and `storage/index/ACTIVE.json` is switched only if
  hit@3 did not drop. Loading an index built with a different embedding model is refused.
  Rollback = point ACTIVE.json at the previous folder.

## 6. Low-data strategy
No model training. Knowledge grows by adding reviewed documents; NLP grows by adding lexicon lines;
safety grows by adding rules + tests. Fine-tuning only becomes worthwhile with thousands of reviewed
Mongolian Q&A pairs **and** an evaluation showing a gap that retrieval + prompting cannot close.

## 7. Use of the provided data (`data/raw/`)
| Dataset | Used for |
|---|---|
| MMRA adverse-drug-reaction reports (999 unique, 2018–2026) | 15 per-drug medication documents (aggregates only), drug-name lexicon, real Mongolian symptom phrasing, anaphylaxis test cases |
| PSUR summaries | not used in MVP (low value for the public) |
| Infectious-disease / mortality statistics | not used in the chatbot (risk of misquoting numbers); usable in project reports |
| Licences, manufacturers, lab statistics | out of scope |
5 byte-identical duplicates and several same-dataset-different-format copies are flagged in `data/raw/manifest.csv`.

## 8. Deviations from the original plan (and why)
| Plan | Implemented | Reason |
|---|---|---|
| ChromaDB | NumPy index behind `KBIndex` | 41 docs; zero native dependencies on student laptops; same interface, swap later |
| bge-m3 embeddings | hashing char-n-gram default, bge-m3 optional | runs offline instantly; Mongolian morphology handled by n-grams + lexicon |
| Streamlit | single-page HTML/JS served by FastAPI | one process to run, nicer demo; React can replace it using the same API |
| SQLAlchemy/PostgreSQL | stdlib `sqlite3` | fewer dependencies; schema is portable to PostgreSQL |

## 9. Default LLM tiering (when a key is configured)
Agent 2 extraction: `claude-haiku-5-5` · Agent 4 answer: `claude-opus-5-5` (effort `medium`, server-side
refusal fallback enabled) · Agent 5 judge: `claude-haiku-5-5`. All configurable in `.env`; another vendor
can be added as one `LLMProvider` subclass in `backend/services/llm.py`.
