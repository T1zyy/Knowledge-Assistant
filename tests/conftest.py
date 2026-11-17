import os
from pathlib import Path
from uuid import uuid4

import httpx
import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from sqlalchemy.engine import make_url

from app.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
ADMIN_KEY = "integration-key-admin-000"
INGEST_KEY = "integration-key-ingest-000"


@pytest.fixture
def settings() -> Settings:
    return Settings(
        database_url="postgresql://unused@localhost/example",
        api_key=ADMIN_KEY,
        ingestion_api_key=INGEST_KEY,
        _env_file=None,
    )


@pytest.fixture
async def running_api(settings, monkeypatch):
    database = os.getenv("TEST_DATABASE_URL")
    redis = os.getenv("TEST_REDIS_URL")
    if not database or not redis:
        pytest.skip("Set TEST_DATABASE_URL and TEST_REDIS_URL for integration tests")
    schema = f"test_{uuid4().hex}"
    admin = psycopg.connect(database, autocommit=True)
    db_name = admin.info.dbname
    if not db_name.endswith("_test"):
        admin.close()
        pytest.fail("Integration tests require a database whose name ends with _test")

    try:
        admin.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        settings.database_url = (
            make_url(database)
            .update_query_dict(
                {
                    "options": f"-csearch_path={schema},public",
                }
            )
            .render_as_string(hide_password=False)
        )
        settings.redis_url = redis
        settings.redis_namespace = f"knowledge-assistant:{schema}"
        monkeypatch.setenv("DATABASE_URL", settings.database_url)
        monkeypatch.setenv("API_KEY", ADMIN_KEY)
        monkeypatch.setenv("INGESTION_API_KEY", INGEST_KEY)
        monkeypatch.setenv("AI_PROVIDER", "demo")
        migration_config = Config(str(ROOT / "alembic.ini"))
        migration_config.attributes["version_table_schema"] = schema
        command.upgrade(migration_config, "head")
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            try:
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://test",
                    headers={"X-API-Key": ADMIN_KEY},
                ) as client:
                    yield client, app.state.knowledge
            finally:
                redis_client = app.state.knowledge.cache.client
                async for key in redis_client.scan_iter(match=f"{settings.redis_namespace}:*"):
                    await redis_client.delete(key)
    finally:
        admin.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(schema)))
        admin.close()
