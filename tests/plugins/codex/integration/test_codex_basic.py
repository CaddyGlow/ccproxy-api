import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from pydantic import TypeAdapter
from tests.helpers.assertions import (
    assert_codex_response_format,
    assert_openai_responses_format,
)
from tests.helpers.test_data import (
    STANDARD_CODEX_REQUEST,
    STANDARD_OPENAI_REQUEST,
)

from ccproxy.llms.models import openai as openai_models
from ccproxy.models.detection import DetectedHeaders, DetectedPrompts
from ccproxy.plugins.codex.models import CodexCacheData


CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"


def _base_response(
    response_id: str,
    *,
    output: list[dict[str, Any]],
    status: str = "completed",
    previous_response_id: str | None = None,
) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1234567890,
        "model": "gpt-5",
        "status": status,
        "parallel_tool_calls": True,
        "previous_response_id": previous_response_id,
        "output": output,
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 15,
        },
    }


def _message_output(text: str, *, item_id: str = "msg_1") -> dict[str, Any]:
    return {
        "type": "message",
        "id": item_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _function_call_output(
    *,
    item_id: str = "call_weather",
    call_id: str = "call_weather",
) -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": "get_weather",
        "arguments": '{"city":"Paris"}',
        "status": "completed",
    }


def _sse_bytes(events: list[dict[str, Any]]) -> bytes:
    chunks = []
    for event in events:
        chunks.append(
            f"event: {event['type']}\n"
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
        )
    chunks.append("data: [DONE]\n\n")
    return "".join(chunks).encode("utf-8")


def _completed_sse(response: dict[str, Any]) -> bytes:
    return _sse_bytes(
        [
            {
                "type": "response.completed",
                "sequence_number": 0,
                "response": response,
            }
        ]
    )


def _function_call_stream_sse(
    response_id: str, *, completed_output: bool = True
) -> bytes:
    function_call = _function_call_output()
    response = _base_response(
        response_id, output=[function_call] if completed_output else []
    )
    return _sse_bytes(
        [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": _base_response(
                    response_id, output=[], status="in_progress"
                ),
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    **function_call,
                    "arguments": "",
                    "status": "in_progress",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "item_id": "call_weather",
                "output_index": 0,
                "delta": '{"city"',
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 3,
                "item_id": "call_weather",
                "output_index": 0,
                "delta": ':"Paris"}',
            },
            {
                "type": "response.function_call_arguments.done",
                "sequence_number": 4,
                "item_id": "call_weather",
                "output_index": 0,
                "arguments": '{"city":"Paris"}',
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 5,
                "output_index": 0,
                "item": function_call,
            },
            {
                "type": "response.completed",
                "sequence_number": 6,
                "response": response,
            },
        ]
    )


def _sse_events(raw_body: bytes) -> list[dict[str, Any]]:
    body = raw_body.decode("utf-8")
    events: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        data_lines = [
            line[6:]
            for line in block.splitlines()
            if line.startswith("data: ") and line[6:] != "[DONE]"
        ]
        if data_lines:
            events.append(json.loads("".join(data_lines)))
    return events


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_models_endpoint_available_when_enabled(
    codex_client: Any,
) -> None:
    """GET /codex/v1/models returns a model list when enabled."""
    resp = await codex_client.get("/codex/v1/models")
    assert resp.status_code == 200
    data: dict[str, Any] = resp.json()
    assert data.get("object") == "list"
    models = data.get("data")
    cli_models = data.get("models")
    assert isinstance(models, list)
    assert len(models) > 0
    assert isinstance(cli_models, list)
    assert len(cli_models) > 0
    assert {"id", "object", "created", "owned_by"}.issubset(models[0].keys())
    assert models[0].get("slug") == models[0]["id"]
    assert models[0].get("display_name") == models[0]["id"]
    assert cli_models[0].get("slug")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_codex_responses_passthrough(
    codex_client: Any,
    mock_external_openai_codex_api: Any,
) -> None:
    """POST /codex/responses proxies to Codex and returns Codex format."""
    resp = await codex_client.post("/codex/responses", json=STANDARD_CODEX_REQUEST)
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    assert_codex_response_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_conversion(
    codex_client: Any,
    mock_external_openai_codex_api: Any,
) -> None:
    """OpenAI /v1/chat/completions converts through Codex and returns OpenAI format."""
    resp = await codex_client.post(
        "/codex/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
    )
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    assert_openai_responses_format(data)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_model_alias_restored_in_response(
    codex_client: Any,
    mock_external_openai_codex_api: Any,
) -> None:
    """Client model aliases stay intact in non-streaming responses."""
    request_payload = {**STANDARD_OPENAI_REQUEST, "model": "gpt-5-nano"}
    resp = await codex_client.post("/codex/v1/chat/completions", json=request_payload)
    assert resp.status_code == 200, resp.text
    data: dict[str, Any] = resp.json()
    assert data.get("model") == "gpt-5"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_chat_completions_streaming(
    codex_client: Any,
    mock_external_openai_codex_api_streaming: Any,
) -> None:
    """Streaming OpenAI /v1/chat/completions returns SSE with valid chunks."""
    # Enable plugin
    request = {**STANDARD_OPENAI_REQUEST, "stream": True}
    resp = await codex_client.post("/codex/v1/chat/completions", json=request)
    raw_body = await resp.aread()

    # Validate SSE headers (note: proxy strips 'connection')
    assert resp.status_code == 200, raw_body
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers.get("cache-control") == "no-cache"

    # Read entire body and split by double newlines to get SSE chunks
    body = raw_body.decode()
    chunks = [c for c in body.split("\n\n") if c.strip()]
    assert chunks, "Expected at least one SSE chunk"
    assert chunks[-1].strip() == "data: [DONE]"
    assert any(chunk.startswith("data: ") for chunk in chunks)


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_codex_bypass_responses_streaming_emits_valid_openai_response_events(
    codex_bypass_client: Any,
) -> None:
    resp = await codex_bypass_client.post(
        "/codex/v1/responses",
        json={
            "model": "gpt-5",
            "stream": True,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reply with exactly OK"}
                    ],
                }
            ],
        },
    )
    raw_body = await resp.aread()

    assert resp.status_code == 200, raw_body
    assert resp.headers["content-type"].startswith("text/event-stream")

    body = raw_body.decode()
    events: list[dict[str, Any]] = []
    validator = TypeAdapter(openai_models.AnyStreamEvent)
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        events.append(event)
        validator.validate_python(event)

    assert events[0]["type"] == "response.created"
    assert events[-1]["type"] == "response.completed"
    assert body.strip().endswith("data: [DONE]")


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_root_openai_responses_paths_work(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response("resp_root", output=[_message_output("OK")])
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response("resp_v1", output=[_message_output("OK")])
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )

    root_resp = await codex_client.post(
        "/responses", json={"model": "gpt-5", "input": "hello"}
    )
    v1_resp = await codex_client.post(
        "/v1/responses", json={"model": "gpt-5", "input": "hello"}
    )

    assert root_resp.status_code == 200, root_resp.text
    assert v1_resp.status_code == 200, v1_resp.text
    assert root_resp.json()["id"] == "resp_root"
    assert v1_resp.json()["id"] == "resp_v1"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_paths_tolerate_stream_options(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    paths = ["/responses", "/v1/responses", "/codex/v1/responses"]
    for index, path in enumerate(paths):
        httpx_mock.add_response(
            url=CODEX_RESPONSES_URL,
            content=_completed_sse(
                _base_response(
                    f"resp_stream_options_{index}", output=[_message_output("OK")]
                )
            ),
            status_code=200,
            headers={"content-type": "text/event-stream"},
        )

        resp = await codex_client.post(
            path,
            json={
                "model": "gpt-5",
                "stream": True,
                "stream_options": {"include_usage": True},
                "prompt_cache_retention": "24h",
                "safety_identifier": "user-123",
                "input": "hello",
            },
        )
        raw_body = await resp.aread()

        assert resp.status_code == 200, raw_body
        events = _sse_events(raw_body)
        assert events[-1]["type"] == "response.completed"
        assert events[-1]["response"]["usage"]["total_tokens"] == 15

    upstream_requests = httpx_mock.get_requests(url=CODEX_RESPONSES_URL)
    assert len(upstream_requests) == len(paths)
    for request in upstream_requests:
        upstream = json.loads(request.content)
        assert upstream["stream"] is True
        assert "stream_options" not in upstream
        assert "prompt_cache_retention" not in upstream
        assert "safety_identifier" not in upstream


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_previous_response_id_tool_loop(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response("resp_tool_1", output=[_function_call_output()])
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response("resp_final", output=[_message_output("Paris is 25C.")])
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )

    headers = {"Authorization": "Bearer client-a"}
    first = await codex_client.post(
        "/responses",
        headers=headers,
        json={
            "model": "gpt-5",
            "input": "What is the weather in Paris?",
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": "required",
        },
    )
    assert first.status_code == 200, first.text
    first_payload = first.json()
    assert first_payload["output"][0]["id"] == "fc_weather"
    assert first_payload["output"][0]["call_id"] == "call_weather"

    second = await codex_client.post(
        "/responses",
        headers=headers,
        json={
            "model": "gpt-5",
            "previous_response_id": "resp_tool_1",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"temperature":"25C"}',
                }
            ],
        },
    )

    assert second.status_code == 200, second.text
    second_payload = second.json()
    assert second_payload["previous_response_id"] == "resp_tool_1"
    assert second_payload["output"][0]["content"][0]["text"] == "Paris is 25C."

    upstream_requests = httpx_mock.get_requests(url=CODEX_RESPONSES_URL)
    assert len(upstream_requests) == 2
    second_upstream = json.loads(upstream_requests[1].content)
    assert "previous_response_id" not in second_upstream
    assert [item["type"] for item in second_upstream["input"]] == [
        "message",
        "function_call",
        "function_call_output",
    ]
    assert second_upstream["input"][1]["id"] == "fc_weather"
    assert second_upstream["input"][1]["call_id"] == "call_weather"
    assert second_upstream["input"][2]["call_id"] == "call_weather"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_streaming_tool_call_ids_and_continuation(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_function_call_stream_sse("resp_stream_tool_1", completed_output=False),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response(
                "resp_stream_final", output=[_message_output("Paris is 25C.")]
            )
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )

    headers = {"Authorization": "Bearer client-a"}
    first = await codex_client.post(
        "/responses",
        headers=headers,
        json={
            "model": "gpt-5",
            "stream": True,
            "stream_options": {"include_usage": True},
            "input": "What is the weather in Paris?",
            "tools": [{"type": "function", "name": "get_weather"}],
            "tool_choice": "required",
        },
    )
    first_body = await first.aread()
    assert first.status_code == 200, first_body
    events = _sse_events(first_body)
    arg_events = [
        event
        for event in events
        if event["type"].startswith("response.function_call_arguments.")
    ]
    assert arg_events
    assert {event["item_id"] for event in arg_events} == {"fc_weather"}
    completed = [event for event in events if event["type"] == "response.completed"][-1]
    assert completed["response"]["output"][0]["id"] == "fc_weather"
    assert completed["response"]["output"][0]["call_id"] == "call_weather"
    assert completed["response"]["usage"]["total_tokens"] == 15

    second = await codex_client.post(
        "/responses",
        headers=headers,
        json={
            "model": "gpt-5",
            "stream": True,
            "stream_options": {"include_usage": True},
            "previous_response_id": "resp_stream_tool_1",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": '{"temperature":"25C"}',
                }
            ],
        },
    )
    second_body = await second.aread()
    assert second.status_code == 200, second_body

    upstream_requests = httpx_mock.get_requests(url=CODEX_RESPONSES_URL)
    first_upstream = json.loads(upstream_requests[0].content)
    assert "stream_options" not in first_upstream
    second_upstream = json.loads(upstream_requests[1].content)
    assert "previous_response_id" not in second_upstream
    assert "stream_options" not in second_upstream
    assert second_upstream["input"][1]["type"] == "function_call"
    assert second_upstream["input"][1]["id"] == "fc_weather"
    assert second_upstream["input"][2]["type"] == "function_call_output"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_unknown_previous_response_id_returns_openai_error(
    codex_client: Any,
) -> None:
    resp = await codex_client.post(
        "/responses",
        json={
            "model": "gpt-5",
            "previous_response_id": "resp_missing",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_missing",
                    "output": "{}",
                }
            ],
        },
    )

    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["param"] == "previous_response_id"
    assert payload["error"]["code"] == "previous_response_not_found"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_provider_detail_error_returns_openai_error(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        json={"detail": "Unsupported parameter: stream_options"},
        status_code=400,
        headers={"content-type": "application/json"},
    )

    resp = await codex_client.post(
        "/responses",
        json={
            "model": "gpt-5",
            "stream": True,
            "input": "hello",
            "extra_rejected_field": True,
        },
    )
    body = await resp.aread()

    assert resp.status_code == 400
    payload = json.loads(body)
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "Unsupported parameter: stream_options"
    assert payload["error"]["param"] == "stream_options"
    assert payload["error"]["code"] == "unsupported_parameter"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_buffered_provider_detail_error_returns_openai_error(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        json={"detail": "Unsupported parameter: stream_options"},
        status_code=400,
        headers={"content-type": "application/json"},
    )

    resp = await codex_client.post(
        "/responses",
        json={
            "model": "gpt-5",
            "input": "hello",
            "extra_rejected_field": True,
        },
    )

    assert resp.status_code == 400
    payload = resp.json()
    assert payload["error"]["type"] == "invalid_request_error"
    assert payload["error"]["message"] == "Unsupported parameter: stream_options"
    assert payload["error"]["param"] == "stream_options"
    assert payload["error"]["code"] == "unsupported_parameter"


@pytest.mark.asyncio
@pytest.mark.integration
@pytest.mark.codex
async def test_openai_responses_state_isolated_by_authorization(
    codex_client: Any,
    httpx_mock: Any,
) -> None:
    httpx_mock.add_response(
        url=CODEX_RESPONSES_URL,
        content=_completed_sse(
            _base_response("resp_client_a", output=[_function_call_output()])
        ),
        status_code=200,
        headers={"content-type": "text/event-stream"},
    )

    first = await codex_client.post(
        "/responses",
        headers={"Authorization": "Bearer client-a"},
        json={"model": "gpt-5", "input": "weather?", "tools": []},
    )
    assert first.status_code == 200, first.text

    second = await codex_client.post(
        "/responses",
        headers={"Authorization": "Bearer client-b"},
        json={
            "model": "gpt-5",
            "previous_response_id": "resp_client_a",
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "call_weather",
                    "output": "{}",
                }
            ],
        },
    )
    assert second.status_code == 400
    assert second.json()["error"]["code"] == "previous_response_not_found"


# Module-scoped client to avoid per-test startup cost
# Use module-level async loop for all tests here
pytestmark = pytest.mark.asyncio(loop_scope="module")


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def codex_client() -> Any:  # type: ignore[misc]
    # Build app and client once per module to avoid factory scope conflicts
    from httpx import ASGITransport, AsyncClient

    from ccproxy.api.app import create_app, initialize_plugins_startup
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        plugins={
            "codex": {"enabled": True},
            "oauth_codex": {"enabled": True},
            "duckdb_storage": {"enabled": False},
            "analytics": {"enabled": False},
            "metrics": {"enabled": False},
        },
        enabled_plugins=["codex", "oauth_codex"],
        plugins_disable_local_discovery=False,  # Enable local plugin discovery
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)

    from ccproxy.plugins.codex.routes import router as codex_router

    app.include_router(codex_router, prefix="/codex")

    credentials_stub = SimpleNamespace(
        access_token="test-codex-access-token",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )
    profile_stub = SimpleNamespace(chatgpt_account_id="test-account-id")

    load_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.load_credentials",
        new=AsyncMock(return_value=credentials_stub),
    )
    profile_patch = patch(
        "ccproxy.plugins.oauth_codex.manager.CodexTokenManager.get_profile_quick",
        new=AsyncMock(return_value=profile_stub),
    )
    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )

    detection_data = CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    detection_patch = patch(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        new=init_detection_stub,
    )
    with load_patch, profile_patch, detection_patch:
        await initialize_plugins_startup(app, settings)

        transport = ASGITransport(app=app)
        runtime = app.state.plugin_registry.get_runtime("codex")
        assert runtime and runtime.adapter, "Codex plugin failed to initialize"
        client = AsyncClient(transport=transport, base_url="http://test")
        try:
            yield client
        finally:
            await client.aclose()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def codex_bypass_client() -> Any:  # type: ignore[misc]
    from httpx import ASGITransport, AsyncClient

    from ccproxy.api.app import create_app, initialize_plugins_startup
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.core import ServerSettings
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    setup_logging(json_logs=False, log_level_name="ERROR")
    settings = Settings(
        enable_plugins=True,
        server=ServerSettings(bypass_mode=True),
        plugins={
            "codex": {"enabled": True},
            "oauth_codex": {"enabled": True},
            "duckdb_storage": {"enabled": False},
            "analytics": {"enabled": False},
            "metrics": {"enabled": False},
        },
        enabled_plugins=["codex", "oauth_codex"],
        plugins_disable_local_discovery=False,
    )
    service_container = create_service_container(settings)
    app = create_app(service_container)

    prompts = DetectedPrompts.from_body(
        {"instructions": "You are a helpful coding assistant."}
    )
    detection_data = CodexCacheData(
        codex_version="fallback",
        headers=DetectedHeaders({}),
        prompts=prompts,
        body_json=prompts.raw,
        method="POST",
        url="https://chatgpt.com/backend-codex/responses",
        path="/api/backend-codex/responses",
        query_params={},
    )

    async def init_detection_stub(self):  # type: ignore[no-untyped-def]
        self._cached_data = detection_data
        return detection_data

    detection_patch = patch(
        "ccproxy.plugins.codex.detection_service.CodexDetectionService.initialize_detection",
        new=init_detection_stub,
    )
    with detection_patch:
        await initialize_plugins_startup(app, settings)

        transport = ASGITransport(app=app)
        runtime = app.state.plugin_registry.get_runtime("codex")
        assert runtime and runtime.adapter, "Codex plugin failed to initialize"
        client = AsyncClient(transport=transport, base_url="http://test")
        try:
            yield client
        finally:
            await client.aclose()
