import json
from dataclasses import dataclass
from uuid import UUID, uuid4

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from app.errors import ServiceError
from app.models import Document, Fragment, Passage


@dataclass(frozen=True)
class IndexState:
    id: UUID
    revision: int
    embedding_space: str | None


@dataclass(frozen=True)
class IndexData:
    filename: str
    media_type: str
    content_hash: str
    external_allowed: bool
    fragments: list[Fragment]
    vectors: list[list[float]]


def missing_document() -> ServiceError:
    return ServiceError(404, "document_not_found", "Document not found")


class Repository:
    def __init__(self, pool: AsyncConnectionPool, embedding_space: str):
        self.pool = pool
        self.embedding_space = embedding_space

    async def state(self, conn: AsyncConnection, *, lock: bool = False) -> IndexState:
        query = "SELECT id, revision, embedding_space FROM corpus WHERE singleton"
        if lock:
            query += " FOR UPDATE"
        row = await (await conn.execute(query)).fetchone()
        return IndexState(**row)

    def check_space(self, state: IndexState) -> None:
        if state.embedding_space not in (None, self.embedding_space):
            raise ServiceError(
                409,
                "embedding_space_mismatch",
                "The corpus was indexed with another embedding provider or model",
            )

    async def current_index(self) -> IndexState:
        async with self.pool.connection() as conn:
            state = await self.state(conn)
        self.check_space(state)
        return state

    async def ready(self, schema_revision: str) -> bool:
        async with self.pool.connection() as conn:
            await self.state(conn)
            row = await (await conn.execute("SELECT version_num FROM alembic_version")).fetchone()
            return bool(row and row["version_num"] == schema_revision)

    async def list_documents(self, limit: int, offset: int) -> list[Document]:
        async with self.pool.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM documents ORDER BY created_at, id LIMIT %s OFFSET %s",
                (limit, offset),
            )
            return [Document.model_validate(row) for row in await cursor.fetchall()]

    async def get_row(self, conn: AsyncConnection, document_id: UUID) -> dict | None:
        return await (
            await conn.execute(
                "SELECT * FROM documents WHERE id = %s",
                (document_id,),
            )
        ).fetchone()

    async def find_hash(self, conn: AsyncConnection, content_hash: str) -> dict | None:
        return await (
            await conn.execute(
                "SELECT * FROM documents WHERE content_hash = %s",
                (content_hash,),
            )
        ).fetchone()

    async def get_document(self, document_id: UUID) -> Document:
        async with self.pool.connection() as conn:
            row = await self.get_row(conn, document_id)
            if row is None:
                raise missing_document()
            return Document.model_validate(row)

    async def check_duplicate(self, content_hash: str) -> Document | None:
        async with self.pool.connection() as conn:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            self.check_space(await self.state(conn))
            row = await self.find_hash(conn, content_hash)
            return Document.model_validate(row) if row else None

    async def write_chunks(self, conn: AsyncConnection, document_id: UUID, data: IndexData) -> None:
        rows = [
            (document_id, ordinal, frag.page, frag.text, json.dumps(vector))
            for ordinal, (frag, vector) in enumerate(
                zip(data.fragments, data.vectors, strict=True), 1
            )
        ]
        async with conn.cursor() as cursor:
            await cursor.executemany(
                """
                INSERT INTO chunks (document_id, ordinal, page, body, embedding)
                VALUES (%s, %s, %s, %s, %s::vector)
                """,
                rows,
            )
        await conn.execute(
            "UPDATE corpus SET revision = revision + 1, embedding_space = %s WHERE singleton",
            (self.embedding_space,),
        )

    async def create(self, data: IndexData) -> tuple[Document, bool]:
        async with self.pool.connection() as conn:
            self.check_space(await self.state(conn, lock=True))
            existing = await self.find_hash(conn, data.content_hash)
            if existing:
                return Document.model_validate(existing), False
            document_id = uuid4()
            row = await (
                await conn.execute(
                    """
                INSERT INTO documents (id, filename, media_type, content_hash, chunk_count,
                                       external_allowed)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING *
                """,
                    (
                        document_id,
                        data.filename,
                        data.media_type,
                        data.content_hash,
                        len(data.fragments),
                        data.external_allowed,
                    ),
                )
            ).fetchone()
            await self.write_chunks(conn, document_id, data)
            return Document.model_validate(row), True

    async def replace(self, current: Document, data: IndexData) -> Document:
        async with self.pool.connection() as conn:
            self.check_space(await self.state(conn, lock=True))
            row = await self.get_row(conn, current.id)
            if row is None:
                raise missing_document()
            if (row["content_hash"], row["filename"], row["external_allowed"]) == (
                data.content_hash,
                data.filename,
                data.external_allowed,
            ):
                return Document.model_validate(row)
            if row["version"] != current.version:
                raise ServiceError(409, "version_conflict", "Document changed during indexing")
            existing = await self.find_hash(conn, data.content_hash)
            if existing and existing["id"] != current.id:
                raise ServiceError(409, "duplicate_content", "Content belongs to another document")
            row = await (
                await conn.execute(
                    """
                UPDATE documents
                SET filename = %s, media_type = %s, content_hash = %s, chunk_count = %s,
                    external_allowed = %s, version = version + 1, updated_at = now()
                WHERE id = %s RETURNING *
                """,
                    (
                        data.filename,
                        data.media_type,
                        data.content_hash,
                        len(data.fragments),
                        data.external_allowed,
                        current.id,
                    ),
                )
            ).fetchone()
            await conn.execute("DELETE FROM chunks WHERE document_id = %s", (current.id,))
            await self.write_chunks(conn, current.id, data)
            return Document.model_validate(row)

    async def delete(self, document_id: UUID) -> None:
        async with self.pool.connection() as conn:
            await self.state(conn, lock=True)
            cursor = await conn.execute("DELETE FROM documents WHERE id = %s", (document_id,))
            if not cursor.rowcount:
                raise missing_document()
            await conn.execute("""
                UPDATE corpus SET revision = revision + 1,
                    embedding_space = CASE WHEN EXISTS (SELECT 1 FROM documents)
                                           THEN embedding_space ELSE NULL END
                WHERE singleton
            """)

    async def retrieve(
        self,
        vector: list[float],
        limit: int,
        *,
        external_only: bool,
    ) -> tuple[IndexState, list[Passage]]:
        async with self.pool.connection() as conn:
            await conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            state = await self.state(conn)
            self.check_space(state)
            cursor = await conn.execute(
                """
                SELECT c.id AS chunk_id, d.id AS document_id, d.filename, d.version,
                       c.ordinal, c.page, c.body AS text,
                       1 - (c.embedding <=> %s::vector) AS similarity
                FROM chunks c JOIN documents d ON d.id = c.document_id
                WHERE NOT %s OR d.external_allowed
                ORDER BY similarity DESC, c.id
                LIMIT %s
                """,
                (json.dumps(vector), external_only, limit),
            )
            return state, [Passage(**row) for row in await cursor.fetchall()]
