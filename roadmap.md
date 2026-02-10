# Roadmap

Last updated: 2026-01-30

## How to use this roadmap

- Update status per item: `Not started` ? `In progress` ? `Blocked` ? `Done`.
- Add notes with date and owner initials.
- Keep scope changes in the Backlog / Ideas section.
- When working on code, feel free to make changes to improve the code and use best practices. 

## Phase 1 - Foundations (config, reproducibility, stability)

| ID | Item | Status | Notes |
|---|---|---|---|
| P1-1 | Centralize config (paths, AWS profiles/regions, KB ID, model ID, K/TOP_K) into one config file and/or env vars | Done | 2026-01-30 BS: Added config.py with env overrides; updated scripts to use it. |
| P1-2 | Add deterministic seeding for testset generation; store seed in outputs | Done | 2026-01-30 BS: Added SEED in config and stored in testset CSV. |
| P1-3 | Fix UTF-8 encoding/mojibake in prompts and strings | Done | 2026-01-30 BS: Normalized Spanish text and emoji encoding in prompts/strings. |
| P1-4 | Improve XML parsing: stricter validation + fallback + logging raw responses | Done | 2026-01-30 BS: Added strict parsing, repair prompt fallback, and raw response logging. |
| P1-5 | Add Bedrock retries with exponential backoff and summarize non-fatal errors | Done | 2026-01-30 BS: Added retry wrappers and run summaries for non-fatal errors. |

## Phase 2 - Evaluation quality & reporting

| ID | Item | Status | Notes |
|---|---|---|---|
| P2-1 | Add cost reporting using token counts and pricing constants | Not started | |
| P2-2 | Improve retrieval evaluation (lexical overlap or embeddings) vs substring containment | Not started | |
| P2-3 | Add metrics: Recall@1, nDCG, average rank; store run metadata | Not started | |
| P2-4 | Add chunking strategy for long KB docs (split or sample) | Not started | |

## Phase 3 - Productization & UX

| ID | Item | Status | Notes |
|---|---|---|---|
| P3-1 | Add basic tests for parsing and metrics | Not started | |
| P3-2 | Add lint/format (ruff/black) and minimal CI/local scripts | Not started | |
| P3-3 | Improve dashboard: run-to-run comparisons, filter by file/source, export filtered data | Not started | |

## Backlog / Ideas

- Provide a CLI wrapper to run the full pipeline with one command
- Add run manifests (JSON) for full reproducibility
- Add dataset versioning and hashing for KB inputs
