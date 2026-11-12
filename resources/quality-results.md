# Quality Results: Hybrid Retrieval

How retrieval changed from the vector-only baseline
(`resources/quality-baseline.md`) to hybrid search. All runs use the same
private 40-case golden set (33 answerable, 7 should refuse), top-k=5, and GPT-4o
STAR synthesis. Only aggregate numbers are published.

## What Changed

1. **Story-level chunks.** The chunker splits on `###` as well as `#`/`##`, and
   every chunk gets a `story_id` (`<file>/<## slug>/<### slug>`). Chunks with no
   text (`---` dividers) are dropped. The corpus went from 15 chunks to 32.
2. **Heading breadcrumbs.** The markdown splitter strips headings, so a story's
   title never reached the embedding. Chunks now start with `H1 > H2 > H3`.
3. **Hybrid retrieval.** A generated `tsvector` column with a GIN index adds
   Postgres full-text search. Vector and keyword results (20 candidates each)
   are merged with reciprocal rank fusion (k=60), which uses only ranks, so the
   two score scales never need normalizing.
4. **Stale-chunk pruning.** Ingest deletes rows for chunks that no longer
   exist, in the same transaction, unless an embedding batch failed.

## Results

Section labels match the original baseline (file plus `##` section). Story
labels are stricter (the exact `###` story).

| Run | Labels | Recall@5 | Unsupported numbers | False refusals | Cost/run |
|---|---|---|---|---|---|
| Baseline, 15 chunks, vector | section | 77.3% | 0 | 2 | $0.37* |
| `###` split, no breadcrumbs, vector | section | 60.6% | 0 | 1 | $0.21* |
| `###` split + breadcrumbs, vector | section | 72.7% | 0 | 1 | $0.23* |
| **Hybrid** | section | **84.8%** | 0 | 1 | $0.28 |
| `###` split + breadcrumbs, vector | story | 56.1% | 0 | 1 | $0.18 |
| **Hybrid** | story | **72.7%** | 0 | 1 | $0.20 |

\* Estimated before cached prompt tokens were priced at the cache rate, so
these overstate cost. Rows without an asterisk use the corrected formula. Cost
also varies run to run with OpenAI's cache hit rate.

Groundedness was 100% and every answer's numbers were found in a cited chunk in
every run.

## Takeaways

- **Hybrid retrieval: +16.6 points story-level recall@5 (56.1% to 72.7%)** for
  about $0.02 more per run, with unsupported numbers still at 0.
- Splitting into smaller chunks hurt recall until headings were put back in
  the chunk text: titles carry most of a story's topic.
- Keyword search alone is weak on these questions (generic words like "time"
  match everything), but fused with vector search it rescues exact-term
  stories.
- The remaining false refusal is a retrieval miss on a question about design
  feedback; story selection with an agent (Q2) is the next lever.

# Q2: Grounding Check and Story-Selection Agent

Same golden set, hybrid retrieval, k=5, GPT-4o. Run on 2026-11-12.

## What Changed

1. **Structured answers with chunk citations.** Synthesis returns JSON
   (`situation`, `task`, `action`, `result`, `citations`, `refused`), parsed
   into a Pydantic model. Citations are chunk IDs, not file names.
2. **Deterministic grounding check.** A graph node checks the draft before
   anything is shown: every cited ID must be a chunk the model was given, and
   every number must appear in a cited chunk. A conditional edge then routes
   to *pass*, *flag* (kept, with a visible warning), or *refuse* (no real
   citation). No LLM judges the answer.
3. **Bounded story-selection agent** (`STORY_SELECTION=agent`). A LangGraph
   subgraph with native tool calls: `search_experiences`, `get_story`,
   `get_story_usage`, `get_company_rubric`, and `select_chunks` to finish.
   It is capped at 6 tool calls and 5 turns. It starts from the pipeline's
   own search for the question, so it can only add stories, and it checks
   which stories were already told to the company.
4. **Story usage tracking.** `interviewkit log` writes to a `story_uses`
   table that the agent reads.

Groundedness is now measured per chunk (cited chunk IDs that were
retrieved), which is stricter than the earlier per-file check.

## Results

| Run | Recall@5 (story) | Groundedness | Unsupported numbers | Refusal accuracy | False refusals | Tool calls | Latency mean | Cost/run |
|---|---|---|---|---|---|---|---|---|
| Hybrid pipeline, text answer (Q1) | 72.7% | 100% (file) | 0 | 97.5% | 1 | - | - | $0.20 |
| **Hybrid pipeline + grounding check** | 72.7% | 100% (chunk) | 0 | **100%** | **0** | - | 3.3s | $0.28 |
| Agent, unseeded | 72.7% | 100% | 0 | 95.0% | 2 | 3.0 | 5.5s | $0.68 |
| **Agent, seeded with pipeline search** | **74.2%** | 100% | 0 | **100%** | **0** | 3.0 | 5.2s | $0.73 |

## Takeaways

- **Structured output fixed the last false refusal.** The JSON `refused` flag
  replaced string matching on the refusal sentence, and refusal accuracy
  went to 100%.
- **The grounding check never fired** (0 flagged, 0 forced refusals) on this
  set, but it is now enforced in code rather than measured after the fact.
- **The first agent was a wash.** Its rewritten queries found 3 stories the
  pipeline missed and lost 4 the pipeline found, and it refused twice.
  Seeding it with the pipeline's search fixed that: +1.5 points recall and no
  false refusals.
- **It isn't worth being the default.** +1.5 points (one case) costs 2.6x
  per run and 1.6x latency, so `prep` keeps the fixed pipeline and the agent
  is opt-in.

## Resume Line

> Built a LangGraph RAG interview-prep CLI over PostgreSQL (pgvector + full-text)
> with a 40-case golden-set eval harness. Hybrid retrieval with reciprocal rank
> fusion raised story-level recall@5 from 56% to 73%, and a deterministic
> grounding check held answers to 0 unsupported metrics and 100% refusal
> accuracy. A bounded tool-calling agent added +1.5 points at 2.6x cost, so the
> pipeline stays the default.
