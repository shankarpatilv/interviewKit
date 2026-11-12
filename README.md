# InterviewKit

![CLI](https://img.shields.io/badge/CLI-Typer-green) ![LangChain](https://img.shields.io/badge/RAG-LangChain-orange) ![LangGraph](https://img.shields.io/badge/Workflow-LangGraph-purple) ![OpenAI](https://img.shields.io/badge/LLM-OpenAI-black) ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue)

A local CLI that turns your own work notes into company-specific behavioral
interview prep: likely questions, plus STAR answers built only from your notes.

## How It Works

**1. You write notes once.** Each job or project gets a markdown file, and each
story in it gets its own `###` heading.

**2. `ingest` makes them searchable.** Every story becomes its own chunk, with
its headings kept in the text so the title counts. Postgres stores each chunk
with two ways to find it: an OpenAI embedding (for meaning) and a full-text
index (for exact words).

**3. `prep` writes the prep.** You give it a job description and a company.
GPT-4o reads the company's principles and writes 5 to 8 likely questions. For
each one, InterviewKit runs two searches: one for chunks that *mean* something
similar, one for chunks that share the *words*. It merges the two lists by
rank, so a story both searches like rises to the top, and hands the best five
to GPT-4o, which writes a STAR answer and cites the exact chunks it used.

**4. Code checks the answer.** Before you see it, InterviewKit checks that
every cited chunk was really retrieved and every number appears in one. A
clean answer passes, a shaky one is shown with a warning, and one with no real
citation is replaced by "not enough context". No LLM grades another LLM.

**5. `eval` keeps it honest.** Questions with known right answers run through
steps 3 and 4, and a scorecard shows whether the right stories came back.

Optional: with `STORY_SELECTION=agent`, a tool-calling agent picks the stories
instead (max 6 tool calls). It can search again, read a whole story, and skip
stories you already told that company (`interviewkit log`).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env              # add OPENAI_API_KEY
docker compose up -d              # Postgres 15 + pgvector

interviewkit init-db
interviewkit ingest               # index experiences/*.md
interviewkit prep --jd jd.txt --company amazon   # also saved to sessions/
```

## Writing Experience Notes

Copy `experiences/_template.md` for each job or project and fill it in: what you
owned, decisions, what broke, tradeoffs, and numbers. Your files are git-ignored
and leave your machine only as embedding and LLM API calls. Company profiles
live in `companies/` (Amazon and Microsoft included).

## Measuring Quality

`interviewkit eval` scores, in code: **recall@k** (was the expected story
retrieved?), **groundedness** (were cited chunks really retrieved?), **metric
fidelity** (is every number in a cited chunk?), **refusal accuracy**, latency,
and cost. Put your golden set in the git-ignored `evals/golden.jsonl` (format:
`evals/golden.example.jsonl`). On 40 cases:

| Setup | Recall@5 | Unsupported numbers | False refusals | Cost/run |
|---|---|---|---|---|
| Meaning-only search | 56% | 0 | 1 | $0.18 |
| Two searches + grounding check (default) | 73% | 0 | 0 | $0.28 |
| Agent (`--selection agent`) | 74% | 0 | 0 | $0.73 |

The agent's extra point isn't worth 2.6x the cost, so it's opt-in ([details](resources/quality-results.md)).

## Tracing (optional)

`docker compose --profile tracing up -d` starts a self-hosted
[Langfuse](https://langfuse.com) to inspect every step, prompt, and token cost.
Set any `pk-lf-...` / `sk-lf-...` pair as `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` in `.env` and open <http://localhost:3000>. Tracing is off
without keys, and self-hosted because traces contain your private notes.

## MCP Server

`interviewkit serve-mcp` exposes `init_database`, `ingest_experiences`, and
`search_experiences` to MCP clients (`--project-dir` is required):

```bash
claude mcp add interviewkit -- /abs/path/.venv/bin/python -m interviewkit \
  serve-mcp --project-dir /abs/path/to/interviewKit
```

## Stack

Python 3.11+, Typer, LangChain, LangGraph, OpenAI (GPT-4o, `text-embedding-3-small`),
PostgreSQL 15 + pgvector, Langfuse, MCP, Pydantic Settings, pytest, Docker Compose.

## What's Next

MCP tools for reading single stories and usage history, plus a `prep_answer`
tool that returns a grounded answer with its check status.
