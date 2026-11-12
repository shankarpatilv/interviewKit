"""Database schema SQL for retrieval storage."""

EMBEDDING_DIMENSION = 1536

CREATE_VECTOR_EXTENSION_SQL = "CREATE EXTENSION IF NOT EXISTS vector"

CREATE_EXPERIENCES_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS experiences (
    id TEXT PRIMARY KEY,
    source_file TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_meta JSONB NOT NULL,
    embedding VECTOR({EMBEDDING_DIMENSION}) NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
)
"""

CREATE_EXPERIENCES_EMBEDDING_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiences_embedding
ON experiences USING ivfflat (embedding vector_cosine_ops)
"""

CREATE_EXPERIENCES_SOURCE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiences_source
ON experiences(source_file)
"""

# Full-text search: a generated tsvector column stays in sync with chunk_text.
ADD_EXPERIENCES_TSV_COLUMN_SQL = """
ALTER TABLE experiences ADD COLUMN IF NOT EXISTS chunk_tsv tsvector
GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED
"""

CREATE_EXPERIENCES_TSV_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_experiences_tsv ON experiences USING GIN (chunk_tsv)
"""

# Which story was told to which company, so the agent can avoid repeats.
CREATE_STORY_USES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS story_uses (
    id BIGSERIAL PRIMARY KEY,
    company TEXT NOT NULL,
    story_id TEXT NOT NULL,
    question TEXT,
    used_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
)
"""

CREATE_STORY_USES_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_story_uses_company_used_at
ON story_uses(company, used_at DESC)
"""

SCHEMA_STATEMENTS = (
    CREATE_VECTOR_EXTENSION_SQL,
    CREATE_EXPERIENCES_TABLE_SQL,
    CREATE_EXPERIENCES_EMBEDDING_INDEX_SQL,
    CREATE_EXPERIENCES_SOURCE_INDEX_SQL,
    ADD_EXPERIENCES_TSV_COLUMN_SQL,
    CREATE_EXPERIENCES_TSV_INDEX_SQL,
    CREATE_STORY_USES_TABLE_SQL,
    CREATE_STORY_USES_INDEX_SQL,
)

UPSERT_CHUNK_SQL = """
INSERT INTO experiences (id, source_file, chunk_text, chunk_meta, embedding)
VALUES (%s, %s, %s, %s::jsonb, %s::vector)
ON CONFLICT (id) DO UPDATE SET
    source_file = EXCLUDED.source_file,
    chunk_text = EXCLUDED.chunk_text,
    chunk_meta = EXCLUDED.chunk_meta,
    embedding = EXCLUDED.embedding
"""

PRUNE_CHUNKS_SQL = """
DELETE FROM experiences WHERE NOT (id = ANY(%s))
"""

SIMILARITY_SEARCH_SQL = """
SELECT source_file, chunk_text, chunk_meta, 1 - (embedding <=> %s::vector) AS similarity, id
FROM experiences
WHERE chunk_meta @> %s::jsonb
ORDER BY (embedding <=> %s::vector) + 0
LIMIT %s
"""

# plainto_tsquery ANDs every word, which rarely matches a full interview
# question; rewriting `&` to `|` ORs the stemmed, stopword-free terms instead.
KEYWORD_SEARCH_SQL = """
WITH q AS (
    SELECT replace(plainto_tsquery('english', %s)::text, ' & ', ' | ')::tsquery AS query
)
SELECT source_file, chunk_text, chunk_meta, ts_rank_cd(chunk_tsv, q.query) AS score, id
FROM experiences, q
WHERE chunk_tsv @@ q.query AND chunk_meta @> %s::jsonb
ORDER BY score DESC, id
LIMIT %s
"""

INSERT_STORY_USE_SQL = """
INSERT INTO story_uses (company, story_id, question) VALUES (%s, %s, %s)
"""

# A NULL company lists uses across all companies.
LIST_STORY_USES_SQL = """
SELECT company, story_id, question, used_at
FROM story_uses
WHERE (%s::text IS NULL OR company = %s)
ORDER BY used_at DESC, id DESC
LIMIT %s
"""

STORY_USE_EXISTS_SQL = """
SELECT EXISTS (SELECT 1 FROM story_uses WHERE company = %s AND story_id = %s)
"""

# Chunk ids are `<file>:<index>:<hash>`, so for one file a shorter id has a
# smaller index; sorting by length first keeps chunk 2 before chunk 10.
GET_STORY_SQL = """
SELECT source_file, chunk_text, chunk_meta, 1.0 AS score, id
FROM experiences
WHERE chunk_meta->>'story_id' = %s
ORDER BY length(id), id
"""
