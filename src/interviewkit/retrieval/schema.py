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

SCHEMA_STATEMENTS = (
    CREATE_VECTOR_EXTENSION_SQL,
    CREATE_EXPERIENCES_TABLE_SQL,
    CREATE_EXPERIENCES_EMBEDDING_INDEX_SQL,
    CREATE_EXPERIENCES_SOURCE_INDEX_SQL,
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

SIMILARITY_SEARCH_SQL = """
SELECT source_file, chunk_text, chunk_meta, 1 - (embedding <=> %s::vector) AS similarity
FROM experiences
ORDER BY (embedding <=> %s::vector) + 0
LIMIT %s
"""
