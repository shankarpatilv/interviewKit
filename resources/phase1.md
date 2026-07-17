# Phase 1 Checkpoint

This checkpoint verifies the local CLI flow:

```text
experience markdown -> chunks -> OpenAI embeddings -> pgvector -> questions -> STAR answers
```

## Prerequisites

- Python 3.11+
- Docker Desktop
- OpenAI API key in `.env`
- private markdown experience files under `experiences/`

The real `.env`, filled-in experience files, and generated session files are
ignored by Git.

## Start Postgres With pgvector

If nothing else is using local port 5432:

```sh
docker compose up -d postgres
docker compose ps
```

If another local Postgres already uses port 5432, map Docker to 5433:

```sh
POSTGRES_PORT=5433 docker compose up -d postgres
POSTGRES_PORT=5433 docker compose ps
```

For the 5433 path, run app commands with a matching `DATABASE_URL`:

```sh
set -a
source .env
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5433/${POSTGRES_DB}"
```

## Initialize And Ingest

Create the schema:

```sh
python -m interviewkit init-db
```

Index private experience documents:

```sh
python -m interviewkit ingest
```

Expected output:

```text
Indexed <n> experience chunks.
```

## Run Prep

Create or choose a job description file, then run:

```sh
python -m interviewkit prep --jd /path/to/job-description.txt --company amazon
```

The command prints markdown to the terminal and writes the same content to:

```text
sessions/<timestamp>-amazon.md
```

Session files are ignored because they may contain private experience details.

## Verified Checkpoint

Verified on 2026-05-20 with Docker pgvector mapped to host port 5433.

Commands:

```sh
POSTGRES_PORT=5433 docker compose up -d postgres
DATABASE_URL=... python -m interviewkit init-db
DATABASE_URL=... python -m interviewkit ingest
DATABASE_URL=... python -m interviewkit prep --jd /tmp/interviewkit-amazon-jd.txt --company amazon
```

Results:

- `init-db` created the pgvector schema.
- `ingest` indexed 19 chunks from two private local experience files.
- `prep` generated 8 Amazon-tagged behavioral questions.
- Each generated question had a STAR answer and at least one source citation.
- Output was written to `sessions/20260520-192104-amazon.md`.

## Troubleshooting

If `init-db` reports that role `interviewkit` does not exist, the app may be
connecting to a different local Postgres on port 5432. Use the 5433 workflow
above and pass the matching `DATABASE_URL`.

If `prep` returns a no-match answer, add more concrete details to the relevant
experience file and rerun `interviewkit ingest`.
