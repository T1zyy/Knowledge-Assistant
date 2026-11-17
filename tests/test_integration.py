import asyncio
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]

VPN_V1 = "Срок действия временного VPN доступа составляет 14 дней."
VPN_V2 = "Срок действия временного VPN доступа составляет 21 день."
VPN_QUESTION = {"question": "Какой срок действия временного VPN доступа?"}


async def upload(client, content=VPN_V1, filename="vpn.txt", mime="text/plain", **kwargs):
    if isinstance(content, str):
        content = content.encode()
    return await client.post("/documents", files={"file": (filename, content, mime)}, **kwargs)


async def test_document_lifecycle_and_cache(running_api):
    client, _ = running_api
    assert (await client.get("/health")).json() == {"status": "ok"}
    assert (await client.get("/ready")).json() == {"status": "ready"}
    assert (await client.post("/ask", json=VPN_QUESTION)).json()["status"] == "insufficient_context"
    saved = await upload(client)
    assert saved.status_code == 201, saved.text
    document = saved.json()
    url = f"/documents/{document['id']}"
    assert (await client.get(url)).json()["version"] == 1

    first = (await client.post("/ask", json=VPN_QUESTION)).json()
    second = (await client.post("/ask", json=VPN_QUESTION)).json()
    assert "14 дней" in first["answer"]
    assert first["sources"][0]["filename"] == "vpn.txt"
    assert not first["cache_hit"]
    assert second["cache_hit"]

    update = await client.put(url, files={"file": ("vpn.txt", VPN_V2.encode(), "text/plain")})
    assert update.json()["version"] == 2
    updated = (await client.post("/ask", json=VPN_QUESTION)).json()
    assert not updated["cache_hit"]
    assert "21 день" in updated["answer"]
    assert updated["sources"][0]["version"] == 2

    assert (await client.delete(url)).status_code == 204
    assert (await client.get(url)).status_code == 404
    assert (await client.post("/ask", json=VPN_QUESTION)).json()["sources"] == []


async def test_concurrent_duplicate_is_idempotent(running_api):
    client, _ = running_api
    first, second = await asyncio.gather(upload(client), upload(client))
    assert sorted([first.status_code, second.status_code]) == [200, 201]
    assert first.json()["id"] == second.json()["id"]
    items = (await client.get("/documents")).json()
    assert len(items["items"]) == 1


async def test_late_answer_cannot_populate_new_revision_cache(running_api, monkeypatch):
    client, knowledge = running_api
    saved = await upload(client)
    url = f"/documents/{saved.json()['id']}"
    entered = asyncio.Event()
    release = asyncio.Event()
    answer = knowledge.provider.answer

    async def paused_answer(question, context):
        if not entered.is_set():
            entered.set()
            await release.wait()
        return await answer(question, context)

    monkeypatch.setattr(knowledge.provider, "answer", paused_answer)
    in_flight = asyncio.create_task(client.post("/ask", json=VPN_QUESTION))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        update = await client.put(url, files={"file": ("vpn.txt", VPN_V2.encode(), "text/plain")})
        assert update.json()["version"] == 2
    finally:
        release.set()
        previous = await in_flight
    assert "14 дней" in previous.json()["answer"]
    current = (await client.post("/ask", json=VPN_QUESTION)).json()
    assert not current["cache_hit"]
    assert "21 день" in current["answer"]


async def test_pdf_and_markdown(running_api):
    client, _ = running_api
    pdf = (ROOT / "examples" / "expenses.pdf").read_bytes()
    assert (await upload(client, pdf, "expenses.pdf", "application/pdf")).status_code == 201
    result = await client.post(
        "/ask", json={"question": "Какой лимит на гостиницу в командировке?"}
    )
    assert "6000" in result.json()["answer"]
    assert result.json()["sources"][0]["page"] == 1
    md = (ROOT / "examples" / "onboarding.md").read_bytes()
    assert (await upload(client, md, "onboarding.md", "text/markdown")).status_code == 201
    listed = (await client.get("/documents?limit=1")).json()
    assert listed["next_offset"] == 1


async def test_failed_replacement_keeps_previous_document(running_api, monkeypatch):
    client, knowledge = running_api
    original = await upload(client)
    url = f"/documents/{original.json()['id']}"

    async def invalid_embedding(_texts):
        return [[0.0]]

    with monkeypatch.context() as patch:
        patch.setattr(knowledge.provider, "embed", invalid_embedding)
        failed = await client.put(url, files={"file": ("vpn.txt", VPN_V2.encode(), "text/plain")})
    assert failed.status_code == 502
    assert (await client.get(url)).json()["version"] == 1
    assert "14 дней" in (await client.post("/ask", json=VPN_QUESTION)).json()["answer"]


async def test_scoped_key_rate_limit_and_safe_errors(running_api):
    client, knowledge = running_api
    settings = knowledge.settings
    settings.ask_requests_per_minute = 1
    limited_key = {"X-API-Key": settings.ingestion_api_key.get_secret_value()}
    assert (await upload(client, headers=limited_key)).status_code == 201
    assert (await client.get("/documents", headers=limited_key)).status_code == 403
    assert (await client.post("/ask", json=VPN_QUESTION, headers=limited_key)).status_code == 403
    assert (await client.get("/documents", headers={"X-API-Key": "bad"})).status_code == 401
    assert (await client.post("/ask", json={"question": "  "})).status_code == 422
    assert (await client.post("/ask", json=VPN_QUESTION)).status_code == 200
    rejected = await client.post("/ask", json=VPN_QUESTION)
    assert rejected.status_code == 429
    assert int(rejected.headers["retry-after"]) > 0
    assert rejected.headers["x-request-id"] == rejected.json()["request_id"]
    assert rejected.headers["cache-control"] == "no-store"


async def test_upload_limits_and_external_consent(running_api, monkeypatch):
    client, knowledge = running_api
    unsupported = await upload(client, "x", "x.exe", "application/octet-stream")
    assert unsupported.status_code == 415
    oversize = await upload(client, b"x" * (knowledge.settings.max_upload_bytes + 1))
    assert oversize.status_code == 413

    async def oversized_stream():
        for _ in range(8):
            yield b"x" * 4096

    streamed = await client.post(
        "/ask",
        content=oversized_stream(),
        headers={"Content-Type": "application/json"},
    )
    assert streamed.status_code == 413

    knowledge.settings.ai_provider = "openai"

    async def never_send(_texts):
        pytest.fail("No data should reach the provider without explicit consent")

    monkeypatch.setattr(knowledge.provider, "embed", never_send)
    assert (await upload(client)).status_code == 403
    assert (await client.post("/ask", json=VPN_QUESTION)).status_code == 403
