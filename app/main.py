import asyncio
import logging
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from psycopg import Error as DatabaseError
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, PoolTimeout
from redis.asyncio import Redis
from redis.exceptions import RedisError
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.ai import DemoProvider, OpenAIProvider
from app.api import router
from app.cache import RedisStore
from app.config import SCHEMA_REVISION, Settings
from app.documents import tokenizer
from app.errors import ServiceError
from app.repository import Repository
from app.service import KnowledgeService

logger = logging.getLogger(__name__)


def error_response(request_id: str, error: ServiceError) -> JSONResponse:
    headers = {}
    if error.retry_after is not None:
        headers["Retry-After"] = str(error.retry_after)
    return JSONResponse(
        {
            "error": {"code": error.code, "message": error.message},
            "request_id": request_id,
        },
        status_code=error.status,
        headers=headers,
    )


class RequestEnvelope:
    def __init__(self, app: ASGIApp, max_upload_bytes: int):
        self.app = app
        self.max_upload_bytes = max_upload_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = uuid4().hex
        scope.setdefault("state", {})["request_id"] = request_id

        async def send_response(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                headers["X-Request-ID"] = request_id
                if scope["path"].startswith(("/documents", "/ask")):
                    headers["Cache-Control"] = "no-store"
            await send(message)

        limit = (
            self.max_upload_bytes + 131_072 if scope["path"].startswith("/documents") else 16_384
        )
        declared = Headers(scope=scope).get("content-length")
        try:
            declared_size = int(declared) if declared is not None else 0
        except ValueError:
            declared_size = -1
        if declared_size < 0:
            response = error_response(
                request_id,
                ServiceError(400, "invalid_content_length", "Invalid Content-Length"),
            )
            await response(scope, receive, send_response)
            return
        if declared_size > limit:
            response = error_response(
                request_id,
                ServiceError(413, "request_too_large", "Request exceeds the size limit"),
            )
            await response(scope, receive, send_response)
            return

        received = 0

        async def bounded_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    raise HTTPException(413, "Request exceeds the size limit")
            return message

        await self.app(scope, bounded_receive, send_response)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await asyncio.to_thread(tokenizer)
        await asyncio.to_thread(tokenizer, "o200k_base")
        pool = AsyncConnectionPool(
            settings.database_url,
            open=False,
            min_size=1,
            max_size=8,
            timeout=3,
            kwargs={"row_factory": dict_row, "connect_timeout": 3},
        )
        redis = Redis.from_url(settings.redis_url, socket_connect_timeout=2, socket_timeout=2)
        openai_key = settings.openai_api_key
        headers = {"Authorization": f"Bearer {openai_key.get_secret_value()}"} if openai_key else {}
        try:
            async with httpx.AsyncClient(
                base_url="https://api.openai.com/v1/",
                headers=headers,
                timeout=settings.openai_timeout_seconds,
            ) as client:
                await pool.open()
                provider = (
                    DemoProvider()
                    if settings.ai_provider == "demo"
                    else OpenAIProvider(
                        settings,
                        client,
                    )
                )
                app.state.knowledge = KnowledgeService(
                    settings,
                    Repository(pool, settings.embedding_space),
                    RedisStore(redis, settings),
                    provider,
                )
                yield
        finally:
            await redis.aclose()
            await pool.close()

    app = FastAPI(title="AI Knowledge Assistant", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(RequestEnvelope, max_upload_bytes=settings.max_upload_bytes)
    app.include_router(router)

    @app.exception_handler(ServiceError)
    async def known_error(request: Request, exc: ServiceError):
        return error_response(request.state.request_id, exc)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _exc: RequestValidationError):
        error = ServiceError(422, "invalid_input", "Check the fields and types in the API schema")
        return error_response(request.state.request_id, error)

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        code = "request_too_large" if exc.status_code == 413 else "http_error"
        return error_response(
            request.state.request_id, ServiceError(exc.status_code, code, str(exc.detail))
        )

    @app.exception_handler(DatabaseError)
    @app.exception_handler(PoolTimeout)
    async def database_error(request: Request, exc: Exception):
        logger.error(
            "Database failure request_id=%s type=%s", request.state.request_id, type(exc).__name__
        )
        return error_response(
            request.state.request_id,
            ServiceError(503, "database_unavailable", "Database is unavailable"),
        )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request) -> dict[str, str]:
        knowledge: KnowledgeService = request.app.state.knowledge
        try:
            schema_ready = await knowledge.repository.ready(SCHEMA_REVISION)
            if schema_ready and await knowledge.cache.client.ping():
                return {"status": "ready"}
        except (DatabaseError, PoolTimeout, RedisError, TypeError):
            pass
        raise ServiceError(503, "not_ready", "Database schema or Redis is unavailable")

    return app
