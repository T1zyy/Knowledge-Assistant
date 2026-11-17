from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated


@dataclass(frozen=True)
class Fragment:
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Passage:
    chunk_id: int
    document_id: UUID
    filename: str
    version: int
    ordinal: int
    page: int | None
    text: str
    similarity: float


class Document(BaseModel):
    id: UUID
    filename: str
    media_type: str
    version: int
    chunk_count: int
    external_allowed: bool
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[Document]
    next_offset: int | None


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1500)
    ]
    allow_external: bool = Field(default=False, strict=True)


class ModelAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["answered", "insufficient_context"]
    answer: str
    source_ids: list[int]


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Source(BaseModel):
    chunk_id: int
    document_id: UUID
    filename: str
    version: int
    fragment: int
    page: int | None
    excerpt: str


class AskResponse(BaseModel):
    status: Literal["answered", "insufficient_context"]
    answer: str
    sources: list[Source]
    cache_hit: bool = False
    usage: Usage | None = None
