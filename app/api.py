from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, Form, Query, Request, Response, Security, UploadFile
from fastapi.security import APIKeyHeader

from app.config import Settings
from app.documents import validate_file
from app.errors import ServiceError
from app.models import AskRequest, AskResponse, Document, DocumentList
from app.service import KnowledgeService

router = APIRouter()
key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass(frozen=True)
class Principal:
    identity: str
    upload_only: bool


def service(request: Request) -> KnowledgeService:
    return request.app.state.knowledge


def authorize(request: Request, key: Annotated[str | None, Security(key_header)]) -> Principal:
    settings: Settings = request.app.state.settings
    header = (key or "").encode()
    if compare_digest(header, settings.api_key.get_secret_value().encode()):
        return Principal(sha256(header).hexdigest(), False)
    if compare_digest(header, settings.ingestion_api_key.get_secret_value().encode()):
        return Principal(sha256(header).hexdigest(), True)
    raise ServiceError(401, "unauthorized", "A valid X-API-Key header is required")


def authorize_admin(principal: Annotated[Principal, Depends(authorize)]) -> Principal:
    if principal.upload_only:
        raise ServiceError(403, "forbidden", "The ingestion key only permits document uploads")
    return principal


async def ingest_file(
    file: UploadFile,
    knowledge: KnowledgeService,
    principal: Principal,
    allow_external: bool,
    document_id: UUID | None = None,
) -> tuple[Document, bool]:
    knowledge.check_external(allow_external)
    await knowledge.cache.enforce(
        principal.identity,
        "upload",
        knowledge.settings.upload_requests_per_minute,
    )
    filename, media_type = validate_file(file.filename, file.content_type)
    size = knowledge.settings.max_upload_bytes
    if file.size is not None and file.size > size:
        raise ServiceError(413, "upload_too_large", "File exceeds the upload limit")
    content = await file.read(size + 1)
    if len(content) > size:
        raise ServiceError(413, "upload_too_large", "File exceeds the upload limit")
    return await knowledge.ingest(content, filename, media_type, allow_external, document_id)


Knowledge = Annotated[KnowledgeService, Depends(service)]
User = Annotated[Principal, Depends(authorize)]
Admin = Annotated[Principal, Depends(authorize_admin)]


@router.post(
    "/documents",
    response_model=Document,
    status_code=201,
    responses={
        200: {"model": Document, "description": "Existing document with identical content"},
    },
)
async def upload_document(
    file: UploadFile,
    response: Response,
    knowledge: Knowledge,
    principal: User,
    allow_external: Annotated[bool, Form()] = False,
) -> Document:
    document, created = await ingest_file(file, knowledge, principal, allow_external)
    if not created:
        response.status_code = 200
    response.headers["Location"] = f"/documents/{document.id}"
    return document


@router.get("/documents", response_model=DocumentList)
async def list_documents(
    knowledge: Knowledge,
    _principal: Admin,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=100_000)] = 0,
) -> DocumentList:
    rows = await knowledge.repository.list_documents(limit + 1, offset)
    return DocumentList(
        items=rows[:limit], next_offset=offset + limit if len(rows) > limit else None
    )


@router.get("/documents/{document_id}", response_model=Document)
async def get_document(document_id: UUID, knowledge: Knowledge, _principal: Admin) -> Document:
    return await knowledge.repository.get_document(document_id)


@router.put("/documents/{document_id}", response_model=Document)
async def replace_document(
    document_id: UUID,
    file: UploadFile,
    knowledge: Knowledge,
    principal: Admin,
    allow_external: Annotated[bool, Form()] = False,
) -> Document:
    document, _created = await ingest_file(file, knowledge, principal, allow_external, document_id)
    return document


@router.delete("/documents/{document_id}", status_code=204)
async def delete_document(document_id: UUID, knowledge: Knowledge, _principal: Admin) -> None:
    await knowledge.repository.delete(document_id)


@router.post("/ask", response_model=AskResponse)
async def ask(payload: AskRequest, knowledge: Knowledge, principal: Admin) -> AskResponse:
    knowledge.check_external(payload.allow_external)
    await knowledge.cache.enforce(
        principal.identity,
        "ask",
        knowledge.settings.ask_requests_per_minute,
    )
    return await knowledge.ask(payload)
