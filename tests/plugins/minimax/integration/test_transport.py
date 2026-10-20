"""Exercise MiniMax routes through the real adapter and mocked HTTP transport."""

import json
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, get_args
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI, Request

from ccproxy.api.bootstrap import create_service_container
from ccproxy.auth.dependencies import get_conditional_auth_manager
from ccproxy.core.logging import get_logger
from ccproxy.core.plugins.hooks import HookManager, HookRegistry
from ccproxy.core.request_context import RequestContext
from ccproxy.models.provider import ModelCard, ModelMappingRule
from ccproxy.plugins.minimax.adapter import MiniMaxAdapter
from ccproxy.plugins.minimax.config import MiniMaxConfig
from ccproxy.plugins.minimax.routes import MiniMaxAdapterDep, MiniMaxConfigDep, router
from ccproxy.services.adapters.format_registry import FormatRegistry
from ccproxy.streaming.handler import StreamingHandler


pytestmark = [pytest.mark.minimax, pytest.mark.integration, pytest.mark.asyncio]


@asynccontextmanager
async def proxy(
    config: MiniMaxConfig, handler: Callable[[httpx.Request], httpx.Response]
) -> AsyncIterator[httpx.AsyncClient]:
    """Only the external HTTP service is faked; route and adapter code are real."""
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as upstream:
        pool = AsyncMock()
        pool.get_client.return_value = upstream
        container = create_service_container()
        adapter = MiniMaxAdapter(
            config=config,
            auth_manager=None,
            http_pool_manager=pool,
            format_registry=container.get_service(FormatRegistry),
            streaming_handler=StreamingHandler(HookManager(HookRegistry())),
        )
        app = FastAPI()
        app.include_router(router, prefix="/minimax")
        app.dependency_overrides[get_args(MiniMaxAdapterDep)[1].dependency] = (
            lambda: adapter
        )
        app.dependency_overrides[get_args(MiniMaxConfigDep)[1].dependency] = (
            lambda: config
        )
        app.dependency_overrides[get_conditional_auth_manager] = lambda: None

        @app.middleware("http")
        async def request_context(request: Request, call_next: Any) -> Any:
            request.state.context = RequestContext(
                request_id="minimax-test",
                start_time=time.perf_counter(),
                logger=get_logger(__name__),
            )
            return await call_next(request)

        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
        ) as client:
            yield client


def reply(request: httpx.Request) -> httpx.Response:
    model = json.loads(request.content)["model"]
    if request.url.path.endswith("/anthropic/v1/messages"):
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "Consider tools",
                        "signature": "opaque",
                    },
                    {
                        "type": "tool_use",
                        "id": "call_test",
                        "name": "lookup",
                        "input": {},
                    },
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )
    return httpx.Response(
        200,
        json={
            "id": "chat_test",
            "object": "chat.completion",
            "created": 1,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )


@pytest.mark.parametrize(
    "region,host", [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]
)
@pytest.mark.parametrize(
    "model,expected",
    [
        ("minimax", "MiniMax-M3.1-Flash"),
        ("MINIMAX-M3.1-FLASH", "MiniMax-M3.1-Flash"),
        ("MiniMax-M3", "MiniMax-M3"),
        ("MiniMax-M2.7", "MiniMax-M2.7"),
        ("saved-custom-model", "saved-custom-model"),
        ("speech-02-hd", "speech-02-hd"),
    ],
)
async def test_chat_model_and_history_passthrough(
    region: str, host: str, model: str, expected: str
) -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return reply(request)

    payload = {
        "model": model,
        "reasoning_effort": "max",
        "thinking": {"type": "adaptive"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.png"},
                    },
                    {"type": "text", "text": "Read this"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Look it up",
                "tool_calls": [
                    {
                        "id": "call_test",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_test", "content": "result"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "lookup", "parameters": {"type": "object"}},
            }
        ],
    }
    config = MiniMaxConfig.model_validate({"api_key": "test-key", "region": region})
    async with proxy(config, upstream) as client:
        result = await client.post("/minimax/v1/chat/completions", json=payload)
    assert result.status_code == 200, result.text
    assert len(captured) == 1
    assert str(captured[0].url) == f"https://{host}/v1/chat/completions"
    assert json.loads(captured[0].content) == {**payload, "model": expected}
    assert captured[0].headers["authorization"] == "Bearer test-key"


@pytest.mark.parametrize(
    "region,host", [("global_en", "api.minimax.io"), ("cn_zh", "api.minimaxi.com")]
)
@pytest.mark.parametrize("model", ["minimax", "MiniMax-M3"])
async def test_native_messages_preserve_full_tool_history(
    region: str, host: str, model: str
) -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return reply(request)

    config = MiniMaxConfig.model_validate({"api_key": "test-key", "region": region})
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 4096,
        "output_config": {"effort": "xhigh"},
        "thinking": {"type": "adaptive"},
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Inspect this"},
                    {
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/image.png",
                        },
                    },
                    {
                        "type": "video",
                        "source": {
                            "type": "url",
                            "url": "https://example.com/video.mp4",
                        },
                    },
                ],
            }
        ],
        "tools": [
            {"name": "lookup", "input_schema": {"type": "object", "properties": {}}}
        ],
    }
    async with proxy(config, upstream) as client:
        first = await client.post("/minimax/v1/messages", json=payload)
        assert first.status_code == 200, first.text
        payload["messages"].extend(
            [
                {"role": "assistant", "content": first.json()["content"]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_test",
                            "content": "result",
                        }
                    ],
                },
            ]
        )
        second = await client.post("/minimax/v1/messages", json=payload)
    assert second.status_code == 200, second.text
    assert len(captured) == 2
    expected = "MiniMax-M3.1-Flash" if model == "minimax" else model
    assert str(captured[-1].url) == f"https://{host}/anthropic/v1/messages"
    assert json.loads(captured[-1].content) == {**payload, "model": expected}
    assert captured[-1].headers["authorization"] == "Bearer test-key"
    assert captured[-1].headers["anthropic-version"] == "2023-06-01"


@pytest.mark.parametrize("path", ["chat/completions", "messages"])
@pytest.mark.parametrize("effort", [None, "low", "medium", "high", "xhigh", "max"])
async def test_flash_effort_and_defaults(path: str, effort: str | None) -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return reply(request)

    payload: dict[str, Any] = {
        "model": "minimax",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if effort is not None:
        if path == "messages":
            payload["output_config"] = {"effort": effort}
        else:
            payload["reasoning_effort"] = effort
    async with proxy(MiniMaxConfig(api_key="test-key"), upstream) as client:
        response = await client.post(f"/minimax/v1/{path}", json=payload)
    assert response.status_code == 200, response.text
    assert captured == [{**payload, "model": "MiniMax-M3.1-Flash"}]


@pytest.mark.parametrize("path", ["chat/completions", "messages"])
@pytest.mark.parametrize("thinking", [None, "adaptive", "disabled"])
async def test_m3_thinking_and_defaults_preserved(
    path: str, thinking: str | None
) -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return reply(request)

    payload: dict[str, Any] = {
        "model": "MiniMax-M3",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    if thinking is not None:
        payload["thinking"] = {"type": thinking}
    async with proxy(MiniMaxConfig(api_key="test-key"), upstream) as client:
        response = await client.post(f"/minimax/v1/{path}", json=payload)
    assert response.status_code == 200
    assert captured == [payload]


@pytest.mark.parametrize("model", ["MiniMax-M3", "MiniMax-M2.7", "saved-custom-model"])
async def test_saved_mapping_keeps_legacy_messages_route(model: str) -> None:
    captured = []

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return reply(request)

    config = MiniMaxConfig(
        api_key="test-key",
        base_url="https://custom.example/v1",
        model_mappings=[ModelMappingRule(match="minimax", target=model)],
    )
    async with proxy(config, upstream) as client:
        response = await client.post(
            "/minimax/v1/messages",
            json={
                "model": "minimax",
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 200, response.text
    assert str(captured[0].url) == "https://custom.example/v1/chat/completions"
    assert json.loads(captured[0].content)["model"] == model


async def test_catalog_and_custom_cards() -> None:
    async with proxy(MiniMaxConfig(api_key="test-key"), reply) as client:
        response = await client.get("/minimax/v1/models")
    assert [card["id"] for card in response.json()["data"]] == [
        "MiniMax-M3.1-Flash",
        "MiniMax-M3",
    ]
    card = ModelCard(id="MiniMax-M2.7", created=1, owned_by="minimax")
    async with proxy(
        MiniMaxConfig(api_key="test-key", models_endpoint=[card]), reply
    ) as client:
        response = await client.get("/minimax/v1/models")
    assert [card["id"] for card in response.json()["data"]] == ["MiniMax-M2.7"]


@pytest.mark.parametrize("path", ["chat/completions", "messages"])
async def test_streaming_transport_and_custom_native_endpoint(path: str) -> None:
    captured = []
    events = [
        {
            "type": "message_start",
            "message": {"id": "msg_stream", "role": "assistant", "content": []},
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "thinking", "thinking": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "Preserved reasoning"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_test",
                "name": "lookup",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": "{}"},
        },
        {"type": "content_block_stop", "index": 1},
        {"type": "message_stop"},
    ]
    stream = "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()
    if path == "chat/completions":
        stream = b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"Preserved reasoning","tool_calls":[{"index":0,"id":"call_test","type":"function","function":{"name":"lookup","arguments":"{}"}}]}}]}\n\ndata: [DONE]\n\n'

    def upstream(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200, content=stream, headers={"content-type": "text/event-stream"}
        )

    config = MiniMaxConfig(
        api_key="test-key", anthropic_base_url="https://custom.example/anthropic"
    )
    payload = {
        "model": "minimax",
        "max_tokens": 4096,
        "stream": True,
        "messages": [{"role": "user", "content": "Hello"}],
    }
    async with proxy(config, upstream) as client:
        response = await client.post(f"/minimax/v1/{path}", json=payload)
    assert response.status_code == 200, response.text
    assert json.loads(captured[0].content) == {**payload, "model": "MiniMax-M3.1-Flash"}
    if path == "messages":
        assert str(captured[0].url) == "https://custom.example/anthropic/v1/messages"
        # The shared SSE handler normalizes JSON whitespace and appends [DONE].
        received = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert received == events
    else:
        assert response.content == stream


async def test_native_upstream_validation_error_is_preserved() -> None:
    error = {
        "error": {
            "type": "invalid_request_error",
            "message": "Flash requires adaptive thinking",
        }
    }

    def upstream(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["thinking"] == {"type": "disabled"}
        return httpx.Response(400, json=error)

    async with proxy(MiniMaxConfig(api_key="test-key"), upstream) as client:
        response = await client.post(
            "/minimax/v1/messages",
            json={
                "model": "minimax",
                "max_tokens": 4096,
                "thinking": {"type": "disabled"},
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )
    assert response.status_code == 400
    assert response.json() == error
