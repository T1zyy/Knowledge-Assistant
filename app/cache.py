import json
import logging
from hashlib import sha256
from uuid import UUID

from pydantic import ValidationError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from app.config import Settings
from app.errors import ServiceError
from app.models import AskResponse

logger = logging.getLogger(__name__)

RATE_LIMIT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('TTL', KEYS[1])}
"""


class RedisStore:
    def __init__(self, client: Redis, settings: Settings):
        self.client = client
        self.settings = settings

    async def enforce(self, identity: str, operation: str, limit: int) -> None:
        key = f"{self.settings.redis_namespace}:limit:{operation}:{identity}"
        try:
            count, ttl = await self.client.eval(RATE_LIMIT, 1, key, 60)
        except RedisError as exc:
            raise ServiceError(
                503, "rate_limiter_unavailable", "Rate limiter is unavailable"
            ) from exc
        if count > limit:
            raise ServiceError(
                429, "rate_limited", "Request limit reached", retry_after=max(1, ttl)
            )

    def answer_key(self, question: str, corpus_id: UUID, revision: int) -> str:
        settings = self.settings
        config = json.dumps(
            [
                "answer-v1",
                settings.embedding_space,
                settings.openai_chat_model,
                settings.retrieval_top_k,
                settings.similarity_threshold,
                settings.context_tokens,
                str(corpus_id),
                revision,
                question,
            ],
            ensure_ascii=False,
        )
        return f"{settings.redis_namespace}:answer:{sha256(config.encode()).hexdigest()}"

    async def get_answer(self, key: str) -> AskResponse | None:
        if not self.settings.cache_ttl_seconds:
            return None
        try:
            value = await self.client.get(key)
            if value:
                return AskResponse.model_validate_json(value).model_copy(
                    update={"cache_hit": True, "usage": None},
                )
        except (RedisError, ValidationError):
            logger.warning("Answer cache read failed")
        return None

    async def store_answer(self, key: str, response: AskResponse) -> None:
        if not self.settings.cache_ttl_seconds:
            return
        try:
            await self.client.set(
                key,
                response.model_dump_json(exclude={"usage"}),
                ex=self.settings.cache_ttl_seconds,
            )
        except RedisError:
            logger.warning("Answer cache write failed")
