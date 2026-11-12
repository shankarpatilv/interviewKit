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
to GPT-4o. GPT-4o writes a STAR answer from those chunks only and cites the
files it used. If nothing fits, it says so instead of making something up.

**4. `eval` keeps it honest.** A set of questions with known right answers runs
through step 3, and a scorecard shows whether the right stories came back and
whether every citation and number in the answers traces to your notes.

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

Copy `experiences/_template.md` for each job or project and fill it in. Answers
are only as good as these notes, so cover what you owned, the decisions you made,
what broke, the tradeoffs, and the numbers. Your filled-in files are git-ignored
and never leave your machine except as embedding and LLM API calls. Company
profiles live in `companies/` (Amazon and Microsoft included).

## Measuring Quality

`interviewkit eval` runs a golden set of questions through retrieval and answer
drafting, then prints a scorecard. Every metric is checked in code, with no
LLM grading another LLM:

- **Recall@k**: was the expected story retrieved?
- **Groundedness**: were all cited sources actually retrieved?
- **Metric fidelity**: does every number in the answer appear in a cited chunk?
- **Refusal accuracy**: did it refuse exactly when there was no matching story?

It also reports latency and cost. Put your golden set in the git-ignored
`evals/golden.jsonl` (format: `evals/golden.example.jsonl`). On 40 cases, the
two-search approach lifted story-level recall@5 from 56% to 73% over meaning-only
search, with 0 unsupported numbers (`resources/quality-results.md`).

## Tracing (optional)

`docker compose --profile tracing up -d` starts a self-hosted
[Langfuse](https://langfuse.com) for inspecting every step, prompt, and token
cost. Set any `pk-lf-...` / `sk-lf-...` pair as `LANGFUSE_PUBLIC_KEY` /
`LANGFUSE_SECRET_KEY` in `.env` and open <http://localhost:3000> (change
`LANGFUSE_PORT` and `LANGFUSE_HOST` if it's taken). Tracing is off without keys,
and it's self-hosted because traces contain your private notes.

## MCP Server

`interviewkit serve-mcp` exposes `init_database`, `ingest_experiences`, and
`search_experiences` to MCP clients like Claude Code:

```bash
claude mcp add interviewkit -- /abs/path/.venv/bin/python -m interviewkit \
  serve-mcp --project-dir /abs/path/to/interviewKit
```

`--project-dir` is required because MCP clients start servers from anywhere.

## Stack

Python 3.11+, Typer, LangChain, LangGraph, OpenAI (GPT-4o, `text-embedding-3-small`;
optional Claude adapter), PostgreSQL 15 + pgvector, Langfuse (self-hosted tracing),
MCP, Pydantic Settings, pytest, Docker Compose.

## What's Next

A bounded agent that picks stories while avoiding ones you've already used, and
a code-level check that every cited chunk and number is real. Each change is
measured against the golden set.
