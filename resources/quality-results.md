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
