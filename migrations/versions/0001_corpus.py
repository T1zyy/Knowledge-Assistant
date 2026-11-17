"""Create the document and vector index tables."""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute("""
        CREATE TABLE corpus (
            singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
            id uuid NOT NULL DEFAULT gen_random_uuid(),
            revision bigint NOT NULL DEFAULT 0,
            embedding_space text
        )
    """)
    op.execute("INSERT INTO corpus DEFAULT VALUES")
    op.execute("""
        CREATE TABLE documents (
            id uuid PRIMARY KEY,
            filename text NOT NULL,
            media_type text NOT NULL,
            content_hash text NOT NULL UNIQUE CHECK (length(content_hash) = 64),
            version integer NOT NULL DEFAULT 1 CHECK (version > 0),
            chunk_count integer NOT NULL CHECK (chunk_count > 0),
            external_allowed boolean NOT NULL,
            created_at timestamptz NOT NULL DEFAULT now(),
            updated_at timestamptz NOT NULL DEFAULT now()
        )
    """)
    op.execute("""
        CREATE TABLE chunks (
            id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
            document_id uuid NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            ordinal integer NOT NULL CHECK (ordinal > 0),
            page integer CHECK (page > 0),
            body text NOT NULL,
            embedding vector(1536) NOT NULL,
            UNIQUE (document_id, ordinal)
        )
    """)


def downgrade() -> None:
    op.execute("DROP TABLE chunks")
    op.execute("DROP TABLE documents")
    op.execute("DROP TABLE corpus")
