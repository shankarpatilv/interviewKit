# Quality Baseline: Vector-Only Retrieval

Baseline scorecard for the current pipeline (pgvector cosine search, top-k=5,
GPT-4o STAR synthesis), recorded before hybrid retrieval and agentic story
selection. The golden set itself stays private because it is built from real
experience documents; only aggregate numbers are published here.

## Setup

- Date: 2026-11-11
- Command: `interviewkit eval --golden evals/golden.jsonl --k 5 --mode vector`
- Golden set: 40 cases (33 expect a story, 7 expect a refusal) across Amazon
  and Microsoft principles
- Corpus: 15 chunks from 2 experience documents
- Labels: expected stories are `source_file` plus `##` section, because chunks
  carry no story ID yet

## Scorecard

| Metric | Value |
|---|---|
| Cases completed | 40/40 |
| Recall@5 | 77.3% |
| Groundedness (citations that were retrieved) | 100.0% |
| Metric fidelity (answers with zero unsupported numbers) | 100.0% |
| Unsupported numbers | 0 |
| Refusal accuracy | 95.0% |
| False refusals | 2 |
| Latency mean / p95 | 2.91s / 4.52s |
| Estimated LLM cost (whole run) | $0.37 |

## Findings

- 8 of 33 answerable cases missed at least one expected story; 6 of those
  target one long section that spans 4 chunks and holds 11 separate technical
  decisions. Semantic search struggles to pick the right chunk for specific
  questions inside that section.
- Both false refusals had recall 0: the model refused correctly given what it
  was handed, so the failure is retrieval, not synthesis.
- Questions with exact technical terms (database choice, design feedback) are
  the misses keyword search should help with, which is what Q1 hybrid
  retrieval targets.
- Metric fidelity is not vacuous: spot-checked answers contained 6 to 7
  numeric claims each, all found in cited chunks. Small integers (1, 2, 3) are
  weak evidence because almost any chunk contains them.

## Fixed Before This Run

- `experiences/README.md` guidance was being indexed as an experience; the
  loader now skips README files, and the 4 stale rows were removed.
- The story graph raised an error when the model correctly refused inside the
  STAR template; refusals are now returned as refusals.

## Known Gaps

- Ingest only upserts, so chunks from deleted or shrunk files stay indexed
  until removed manually.
- Section-level labels are coarse; Q1 story IDs make them precise.
