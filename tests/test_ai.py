import json
from uuid import uuid4

import httpx
import pytest

from app.ai import OpenAIProvider, metadata_for
from app.errors import ServiceError
from app.models import ModelAnswer, Passage
from app.service import grounded_answer


@pytest.fixture
def passage() -> Passage:
    return Passage(4, uuid4(), "vpn.pdf", 2, 1, 3, "Доступ выдаётся на 14 дней.", 0.9)


async def test_structured_output_and_tool_round_trip(settings, passage):
    calls = []

    def mock_openai(request: httpx.Request):
        payload = json.loads(request.content)
        calls.append(payload)
        assert request.url.path == "/v1/responses"
        assert payload["store"] is False
        assert payload["text"]["format"]["strict"] is True
        if len(calls) == 1:
            assert payload["tool_choice"] == "auto"
            return httpx.Response(
                200,
                json={
                    "output": [
                        {
                            "type": "function_call",
                            "name": "get_document_metadata",
                            "arguments": '{"chunk_id":4}',
                            "call_id": "call_123",
                        }
                    ],
                    "usage": {"input_tokens": 22, "output_tokens": 12},
                },
            )
        assert payload["tool_choice"] == "none"
        assert payload["input"][-1]["call_id"] == "call_123"
        assert json.loads(payload["input"][-1]["output"]) == {
            "filename": "vpn.pdf",
            "page": 3,
            "version": 2,
        }
        result = {"status": "answered", "answer": "Версия 2.", "source_ids": [4]}
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": json.dumps(result)}],
                    }
                ],
                "usage": {"input_tokens": 38, "output_tokens": 19},
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(mock_openai),
        base_url="https://api.openai.com/v1/",
    ) as client:
        draft, usage = await OpenAIProvider(settings, client).answer("Какая версия?", [passage])
    answer = grounded_answer(draft, [passage], usage)
    assert answer.sources[0].page == 3
    assert answer.sources[0].excerpt == passage.text
    assert answer.usage.input_tokens == 60
    assert len(calls) == 2


def test_unknown_source_is_not_resolved(passage):
    call = {"name": "get_document_metadata", "arguments": '{"chunk_id":99}'}
    assert metadata_for(call, {4: passage}) == {"error": "unknown_source"}
    raw = ModelAnswer(status="answered", answer="Наверное.", source_ids=[4, 99])
    assert grounded_answer(raw, [passage], None).status == "insufficient_context"


async def test_embeddings_sorted_by_index(settings):
    def handler(request):
        assert request.url.path == "/v1/embeddings"
        payload = json.loads(request.content)
        assert payload["input"] == ["a", "b"]
        assert payload["dimensions"] == 1536
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [2.0]},
                    {"index": 0, "embedding": [1.0]},
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1/",
    ) as client:
        assert await OpenAIProvider(settings, client).embed(["a", "b"]) == [[1.0], [2.0]]


@pytest.mark.parametrize(("status", "expect"), [(401, 502), (429, 503)])
async def test_non_retryable_provider_errors(settings, status, expect):
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(status, json={"error": {"code": "insufficient_quota"}})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1/",
    ) as client:
        with pytest.raises(ServiceError) as exc:
            await OpenAIProvider(settings, client).embed(["a"])
    assert attempts == 1
    assert exc.value.status == expect


async def test_timeout_retries_are_bounded(settings, monkeypatch):
    attempts = 0

    def handler(request):
        nonlocal attempts
        attempts += 1
        raise httpx.ReadTimeout("test timeout", request=request)

    async def skip_delay(_seconds):
        return None

    monkeypatch.setattr("app.ai.asyncio.sleep", skip_delay)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.openai.com/v1/",
    ) as client:
        with pytest.raises(ServiceError) as exc:
            await OpenAIProvider(settings, client).embed(["a"])
    assert attempts == 3
    assert exc.value.status == 504
