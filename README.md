# InterviewKit

InterviewKit is a local CLI for the part of interview prep that is easy to
avoid until it suddenly matters: turning your own messy work history into
clear, company-specific behavioral stories.

You write markdown notes about real projects you have worked on. InterviewKit
indexes those notes with OpenAI embeddings, stores them in PostgreSQL with
pgvector, retrieves the stories that match a role, and drafts behavioral
questions and STAR answers with citations back to the source files.

## What It Does

- Loads private markdown experience notes from a local `experiences/` folder.
- Splits experience documents into retrieval-friendly chunks while preserving
  source-file and markdown-header metadata.
- Generates OpenAI `text-embedding-3-small` embeddings for experience chunks.
- Stores embedded chunks in PostgreSQL 15 with pgvector for semantic search.
- Generates 5 to 8 company-specific behavioral interview questions from a job
  description and company principles.
- Retrieves relevant experience chunks for each question.
- Drafts STAR-format answers using only retrieved experience context.
- Cites the source experience file used for each generated answer.
- Writes prep output to both the terminal and a markdown session file.

## Why It Exists

Behavioral prep is weirdly painful. You usually have the stories, but they are
scattered across memory, notes, brag docs, tickets, and half-remembered
projects. Then every company asks for the same raw material through a different
lens: Amazon Leadership Principles, Microsoft competencies, startup values,
role-specific expectations.

InterviewKit makes that process less ad hoc. Write the experience notes once,
then reuse them. The tool finds the relevant parts, drafts questions you are
likely to hear, and turns the right stories into cited STAR answers.

## How It Works

```text
experience markdown
    -> LangChain loading and markdown-aware chunking
    -> OpenAI embeddings
    -> PostgreSQL + pgvector storage
    -> LangChain semantic retrieval
    -> LangGraph question and answer workflows
    -> cited interview-prep markdown output
```

The experience notes are the important part. Each file should describe a real
project: what happened, what you owned, what decisions you made, what broke,
what tradeoffs mattered, and what changed because of the work. InterviewKit
does not try to invent a better career for you. It searches those notes and
uses the matching chunks as evidence.

## Core Features

### Experience Ingestion

`interviewkit ingest` reads markdown files from `experiences/`, skips
templates, splits the useful text into chunks, embeds those chunks with
OpenAI, and upserts them into the local pgvector table.

### Company-Aware Questions

`interviewkit prep` reads a job description and a company profile from
`companies/`, then generates likely behavioral questions tagged with the
principle or competency they are testing.

### Retrieval-Grounded STAR Answers

For each generated question, InterviewKit searches pgvector for relevant
experience chunks and drafts an answer in STAR format: Situation, Task, Action,
Result, and Sources.

If the retrieved context is not good enough, it says so instead of making up a
story. That is deliberate. A confident fake answer is worse than no answer.

### Source Citations

Generated answers cite source files in this format:

```text
(source: amazon-bedrock-pipeline.md)
```

This keeps the generated prep tied to actual notes instead of floating around
as generic AI advice.

### Local Templates

The repository includes public templates for company profiles and experience
documents. Filled-in personal experience files are ignored by Git. Your private
career details should stay private.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| CLI | Typer |
| Document and retrieval plumbing | LangChain |
| Workflow orchestration | LangGraph |
| LLM provider | OpenAI behind a provider abstraction |
| Optional provider adapter | Claude wrapper |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | PostgreSQL 15 with pgvector |
| Configuration | Pydantic Settings |
| Testing | pytest, Ruff, mypy, Black |
| Local services | Docker Compose |

## Example Workflow

```bash
# Install locally
python -m pip install -e '.[dev]'

# Start PostgreSQL with pgvector
docker compose up -d

# Initialize the database schema
python -m interviewkit init-db

# Index private experience documents
python -m interviewkit ingest

# Generate prep for a role and company
python -m interviewkit prep --jd path/to/job-description.txt --company amazon
```

Session output is written under `sessions/`, which is ignored by Git.

## Repository Layout

```text
src/interviewkit/
  agent/        LangGraph workflows for question and answer generation
  ingest/       markdown loading, chunking, and embedding
  providers/    OpenAI-first provider abstraction and optional Claude adapter
  retrieval/    PostgreSQL/pgvector schema, storage, search, and retriever
  prompts/      prompt templates loaded at runtime
  cli.py        Typer command entry point
  config.py     Pydantic settings

companies/      company principles and interview rubrics
examples/       public examples and templates
experiences/    local private experience notes, ignored by Git
resources/      local usage and checkpoint notes
sessions/       generated prep outputs, ignored by Git
tests/          pytest test suite
```

## Design Principles

- Keep private career details local.
- Ground answers in retrieved notes, not vibes.
- Preserve `source_file` metadata all the way to the final citation.
- Use explicit LangGraph workflows instead of loose autonomous agent loops.
- Keep CLI commands thin; put real behavior in library code.
- Prefer boring, typed Python over clever framework magic.
- Test LLM and retrieval behavior with fake providers where possible.

## Status

InterviewKit provides a local workflow for behavioral interview prep: ingest
private experience notes, generate company-specific questions, retrieve
relevant stories, draft STAR answers, and cite source files.

Story reuse tracking, multi-turn mock interviews, richer evaluation, and answer
quality scoring are planned extensions beyond this core CLI workflow.
