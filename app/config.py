from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

EMBEDDING_DIMENSIONS = 1536
SCHEMA_REVISION = "0001"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    database_url: str
    redis_url: str = "redis://localhost:6379/0"
    api_key: SecretStr
    ingestion_api_key: SecretStr

    ai_provider: Literal["demo", "openai"] = "demo"
    allow_external_ai: bool = False
    openai_api_key: SecretStr | None = None
    openai_chat_model: str = Field(default="gpt-4.1-mini", min_length=1)
    openai_embedding_model: str = Field(default="text-embedding-3-small", min_length=1)
    openai_timeout_seconds: float = Field(default=12, gt=0, le=30)

    max_upload_bytes: int = Field(default=2 * 1024 * 1024, ge=256, le=10 * 1024 * 1024)
    max_pdf_pages: int = Field(default=40, ge=1, le=100)
    max_chunks: int = Field(default=100, ge=1, le=500)
    chunk_tokens: int = Field(default=500, ge=32, le=2000)
    chunk_overlap: int = Field(default=64, ge=0)
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    context_tokens: int = Field(default=3000, ge=500, le=10000)
    min_similarity: float | None = Field(default=None, ge=0, le=1)

    cache_ttl_seconds: int = Field(default=300, ge=0, le=3600)
    ask_requests_per_minute: int = Field(default=20, ge=1)
    upload_requests_per_minute: int = Field(default=8, ge=1)
    redis_namespace: str = Field(default="knowledge-assistant", pattern=r"^[a-zA-Z0-9:_-]+$")

    @model_validator(mode="after")
    def validate_configuration(self) -> "Settings":
        api_key = self.api_key.get_secret_value()
        ingest_key = self.ingestion_api_key.get_secret_value()
        if min(len(api_key), len(ingest_key)) < 16 or api_key == ingest_key:
            raise ValueError("API keys must be distinct and at least 16 characters")
        if self.chunk_overlap >= self.chunk_tokens:
            raise ValueError("Chunk overlap must be smaller than the chunk size")
        if self.ai_provider == "openai" and (
            not self.allow_external_ai
            or not self.openai_api_key
            or not self.openai_api_key.get_secret_value().strip()
        ):
            raise ValueError("OpenAI mode requires ALLOW_EXTERNAL_AI and OPENAI_API_KEY")
        return self

    @property
    def embedding_space(self) -> str:
        if self.ai_provider == "demo":
            return "demo:feature-hash-v1"
        return f"openai:{self.openai_embedding_model}:{EMBEDDING_DIMENSIONS}"

    @property
    def similarity_threshold(self) -> float:
        if self.min_similarity is not None:
            return self.min_similarity
        return 0.12 if self.ai_provider == "demo" else 0.35
