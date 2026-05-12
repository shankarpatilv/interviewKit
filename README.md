# InterviewKit

InterviewKit is a local AI interview-prep CLI that turns a candidate's own experience notes into company-specific behavioral interview practice.

It reads markdown experience documents, indexes them with embeddings, retrieves the most relevant stories for a target role, and generates tailored interview questions plus STAR-format draft answers with citations back to the source experiences. It is designed for candidates preparing across multiple interview loops where each company has its own values, competencies, and expectations.

## What It Does

- Generates likely behavioral interview questions from a job description and company principles.
- Synthesizes STAR-format draft answers from the candidate's own experience corpus.
- Cites the experience documents used for each answer so drafts stay grounded.
- Tracks which stories have already been used for a company or interview round.
- Supports live mock interview flows for multi-turn practice and feedback.
- Runs locally as a CLI so private experience documents stay on the user's machine.

## Why It Exists

Behavioral interview prep is repetitive but high-stakes. Candidates often spend hours mapping the same set of projects, conflicts, tradeoffs, and wins to different company rubrics: Amazon Leadership Principles, Microsoft competencies, startup founder values, or role-specific expectations.

InterviewKit makes that workflow faster and more systematic. The candidate writes rich experience notes once, then reuses them through retrieval-augmented generation to prepare for each company with better coverage and less manual rewriting.

## How It Works

```text
experience markdown
    -> LangChain loading and chunking
    -> OpenAI embeddings
    -> PostgreSQL + pgvector
    -> LangGraph interview-prep workflow
    -> tailored questions, cited STAR answers, and session output
```

The experience corpus is the core of the system. Each document captures the context, role, decisions, tradeoffs, metrics, conflict, and lessons from a real project or work experience. InterviewKit retrieves relevant chunks from that corpus and uses them to draft answers for the target company and role.

## Core Features

### Experience Ingestion

InterviewKit loads markdown files from an experience directory, splits them into retrieval-friendly chunks, embeds them, and stores them in PostgreSQL with pgvector.

### Company-Aware Prep

Given a job description and a company profile, InterviewKit identifies likely behavioral themes and maps them to relevant experience stories.

### STAR Answer Drafting

Generated answers follow the STAR structure: Situation, Task, Action, Result. The answers are grounded in retrieved experience chunks and include citations so users can trace each answer back to source notes.

### Story Usage Tracking

InterviewKit records which stories were used for which companies and rounds. This helps avoid repeating the same story too often in follow-up interviews.

### Mock Interview Practice

The mock interview flow uses a multi-turn LangGraph workflow to play the role of an interviewer, ask follow-up questions, and produce feedback after the session.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Language | Python 3.11+ |
| CLI | Typer |
| RAG pipeline | LangChain |
| Agent orchestration | LangGraph |
| LLM providers | Anthropic Claude and OpenAI |
| Embeddings | OpenAI `text-embedding-3-small` |
| Vector database | PostgreSQL 15 with pgvector |
| Configuration | Pydantic Settings |
| Testing | pytest |
| Local services | Docker Compose |

## Example Workflow

```bash
# Install locally
pip install -e .

# Start PostgreSQL with pgvector
docker compose up -d

# Index experience documents
python -m interviewkit ingest

# Generate interview prep for a role
python -m interviewkit prep --jd path/to/job-description.txt --company amazon

# Log a story after using it in an interview
python -m interviewkit log --company amazon --story bedrock-pipeline

# Run a mock interview
python -m interviewkit mock --jd path/to/job-description.txt --company amazon
```

## Repository Layout

```text
src/interviewkit/
  agent/        LangGraph workflows for prep, story tracking, and mock interviews
  ingest/       markdown loading, chunking, and embedding
  providers/    Anthropic and OpenAI provider adapters
  retrieval/    pgvector storage and retrieval
  prompts/      prompt templates loaded at runtime
  cli.py        Typer command entry point
  config.py     Pydantic settings

companies/      company principles and interview rubrics
examples/       public templates for user-owned private files
evals/          manual quality evaluation data
tests/          pytest test suite
```

Private experience documents and generated sessions are intentionally kept out of Git.

## Design Principles

- Keep private career details local.
- Ground generated answers in cited experience notes.
- Use explicit LangGraph workflows instead of opaque agent loops.
- Keep CLI commands thin and business logic in library modules.
- Prefer simple, typed Python over framework-heavy magic.
- Treat automated scoring as future work; answer quality should be reviewed by the user.

## Status

InterviewKit is designed as a complete local AI interview-prep system for behavioral interviews. The public repository is structured to show the product architecture, setup flow, and intended user experience while keeping personal experience data private.
