import asyncio
import hashlib
import json
import math
from uuid import UUID

from app.ai import AIProvider, invalid_provider_response
from app.cache import RedisStore
from app.config import EMBEDDING_DIMENSIONS, Settings
from app.documents import split, tokenizer
from app.errors import ServiceError
from app.models import AskRequest, AskResponse, Document, ModelAnswer, Passage, Source, Usage
from app.repository import IndexData, Repository


def validate_vectors(vectors: list[list[float]], count: int) -> None:
    if not isinstance(vectors, list) or len(vectors) != count:
        raise invalid_provider_response()
    for vector in vectors:
        if (
            not isinstance(vector, list)
            or len(vector) != EMBEDDING_DIMENSIONS
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
            or not any(vector)
        ):
            raise invalid_provider_response()


def insufficient_context(usage: Usage | None = None) -> AskResponse:
    return AskResponse(
        status="insufficient_context",
        answer="В загруженных документах недостаточно информации для ответа.",
        sources=[],
        usage=usage,
    )


def grounded_answer(
    model: ModelAnswer,
    passages: list[Passage],
    usage: Usage | None,
) -> AskResponse:
    available = {passage.chunk_id: passage for passage in passages}
    ids = list(dict.fromkeys(model.source_ids))
    if (
        model.status != "answered"
        or not model.answer.strip()
        or not ids
        or any(chunk_id not in available for chunk_id in ids)
    ):
        return insufficient_context(usage)
    sources = []
    for chunk_id in ids:
        passage = available[chunk_id]
        sources.append(
            Source(
                chunk_id=chunk_id,
                document_id=passage.document_id,
                filename=passage.filename,
                version=passage.version,
                fragment=passage.ordinal,
                page=passage.page,
                excerpt=passage.text[:360],
            )
        )
    return AskResponse(status="answered", answer=model.answer.strip(), sources=sources, usage=usage)


class KnowledgeService:
    def __init__(
        self,
        settings: Settings,
        repository: Repository,
        cache: RedisStore,
        provider: AIProvider,
    ):
        self.settings = settings
        self.repository = repository
        self.cache = cache
        self.provider = provider

    def check_external(self, allowed: bool) -> None:
        if self.settings.ai_provider == "openai" and not allowed:
            raise ServiceError(
                403,
                "external_processing_not_approved",
                "Set allow_external=true only for data approved for external processing",
            )

    async def ingest(
        self,
        content: bytes,
        filename: str,
        media_type: str,
        external_allowed: bool,
        document_id: UUID | None = None,
    ) -> tuple[Document, bool]:
        self.check_external(external_allowed)
        digest = hashlib.sha256(content).hexdigest()
        current = None
        if document_id is None:
            duplicate = await self.repository.check_duplicate(digest)
            if duplicate:
                return duplicate, False
        else:
            await self.repository.current_index()
            current = await self.repository.get_document(document_id)
        try:
            async with asyncio.timeout(75):
                fragments = await asyncio.to_thread(split, content, media_type, self.settings)
                vectors = await self.provider.embed([chunk.text for chunk in fragments])
                validate_vectors(vectors, len(fragments))
        except TimeoutError as exc:
            raise ServiceError(504, "indexing_timeout", "Document indexing timed out") from exc
        data = IndexData(filename, media_type, digest, external_allowed, fragments, vectors)
        if current is not None:
            return await self.repository.replace(current, data), False
        return await self.repository.create(data)

    def select_context(self, passages: list[Passage]) -> list[Passage]:
        selected = []
        seen = set()
        budget = self.settings.context_tokens
        encoding = tokenizer("o200k_base")
        for passage in passages:
            if passage.similarity < self.settings.similarity_threshold or passage.text in seen:
                continue
            payload = json.dumps({"id": passage.chunk_id, "text": passage.text}, ensure_ascii=False)
            cost = len(encoding.encode_ordinary(payload)) + 4
            if cost > budget:
                continue
            budget -= cost
            seen.add(passage.text)
            selected.append(passage)
        return selected

    async def ask(self, request: AskRequest) -> AskResponse:
        self.check_external(request.allow_external)
        state = await self.repository.current_index()
        if state.embedding_space is None:
            return insufficient_context()
        key = self.cache.answer_key(request.question, state.id, state.revision)
        cached = await self.cache.get_answer(key)
        if cached is not None:
            return cached
        try:
            async with asyncio.timeout(45):
                vectors = await self.provider.embed([request.question])
                validate_vectors(vectors, 1)
                snapshot, results = await self.repository.retrieve(
                    vectors[0],
                    self.settings.retrieval_top_k,
                    external_only=self.settings.ai_provider == "openai",
                )
                context = self.select_context(results)
                if not context:
                    return insufficient_context()
                result, usage = await self.provider.answer(request.question, context)
        except TimeoutError as exc:
            raise ServiceError(504, "answer_timeout", "Question processing timed out") from exc

        response = grounded_answer(result, context, usage)
        if response.status == "answered":
            key = self.cache.answer_key(request.question, snapshot.id, snapshot.revision)
            await self.cache.store_answer(key, response)
        return response
