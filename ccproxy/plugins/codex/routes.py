"""Codex plugin routes."""

import json
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocketState

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import (
    get_plugin_adapter,
    get_provider_config_dependency,
)
from ccproxy.auth.dependencies import ConditionalAuthDep
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
    UPSTREAM_ENDPOINT_OPENAI_RESPONSES,
)
from ccproxy.core.plugins import PluginRegistry, ProviderPluginRuntime
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.sse_parser import SSEStreamParser

from .config import CodexSettings


if TYPE_CHECKING:
    pass

CodexAdapterDep = Annotated[Any, Depends(get_plugin_adapter("codex"))]
CodexConfigDep = Annotated[
    CodexSettings,
    Depends(get_provider_config_dependency("codex", CodexSettings)),
]
router = APIRouter()


# Helper to handle adapter requests
async def handle_codex_request(
    request: Request,
    adapter: Any,
) -> StreamingResponse | Response | DeferredStreaming:
    result = await adapter.handle_request(request)
    return cast(StreamingResponse | Response | DeferredStreaming, result)


# Route definitions
async def _codex_responses_handler(
    request: Request,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    """Shared handler for Codex responses endpoints."""

    return await handle_codex_request(request, adapter)


def _get_codex_websocket_adapter(websocket: WebSocket) -> Any:
    if not hasattr(websocket.app.state, "plugin_registry"):
        raise RuntimeError("Plugin registry not initialized")

    registry: PluginRegistry = websocket.app.state.plugin_registry
    runtime = registry.get_runtime("codex")

    if not runtime or not isinstance(runtime, ProviderPluginRuntime):
        raise RuntimeError("Codex plugin not initialized")

    if not runtime.adapter:
        raise RuntimeError("Codex adapter not available")

    return runtime.adapter


def _prepare_websocket_headers(websocket: WebSocket) -> dict[str, str]:
    headers = {
        key.lower(): value
        for key, value in websocket.headers.items()
        if not key.lower().startswith("sec-websocket-")
    }
    headers["accept"] = "text/event-stream"
    return headers


def _parse_websocket_request(raw_message: str) -> dict[str, Any]:
    payload = json.loads(raw_message)
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object payload")

    if payload.get("type") != "response.create":
        raise ValueError("Unsupported websocket message type")

    provider_payload = dict(payload)
    provider_payload.pop("type", None)
    return provider_payload


def _make_websocket_terminal_event(
    provider_payload: dict[str, Any],
    *,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    response_payload = {
        "id": f"resp_ws_{uuid4().hex}",
        "object": "response",
        "created_at": int(time()),
        "status": "failed" if error else "completed",
        "model": provider_payload.get("model"),
        "output": [],
        "parallel_tool_calls": False,
        "error": error,
        "incomplete_details": None,
    }
    return {"type": "response.completed", "response": response_payload}


def _is_websocket_warmup_request(provider_payload: dict[str, Any]) -> bool:
    input_items = provider_payload.get("input")
    return isinstance(input_items, list) and len(input_items) == 0


def _serialize_codex_models(config: CodexSettings) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for card in config.models_endpoint:
        model_data = card.model_dump(mode="json")
        slug = model_data.get("slug") or model_data.get("id") or model_data.get("root")
        if isinstance(slug, str) and slug:
            model_data.setdefault("slug", slug)
            model_data.setdefault("display_name", slug)
        models.append(model_data)
    return models


def _load_codex_cli_models_cache() -> list[dict[str, Any]]:
    cache_file = Path.home() / ".codex" / "models_cache.json"
    if not cache_file.exists():
        return []

    try:
        payload = json.loads(cache_file.read_text())
    except Exception:
        return []

    models = payload.get("models")
    if not isinstance(models, list):
        return []

    return [model for model in models if isinstance(model, dict)]


def _serialize_codex_cli_models(config: CodexSettings) -> list[dict[str, Any]]:
    configured_ids = {
        card.id for card in config.models_endpoint if isinstance(getattr(card, "id", None), str)
    }
    configured_ids.update(
        {
            card.root
            for card in config.models_endpoint
            if isinstance(getattr(card, "root", None), str) and card.root
        }
    )

    cached_models = _load_codex_cli_models_cache()
    if cached_models and configured_ids:
        matched = [
            model
            for model in cached_models
            if model.get("slug") in configured_ids or model.get("display_name") in configured_ids
        ]
        if matched:
            return matched

    return _serialize_codex_models(config)


async def _stream_websocket_response(
    websocket: WebSocket,
    adapter: Any,
    provider_payload: dict[str, Any],
) -> None:
    request_headers = _prepare_websocket_headers(websocket)
    provider_payload["stream"] = True
    provider_payload["store"] = False
    provider_headers = await adapter.prepare_provider_headers(request_headers)
    target_url = await adapter.get_target_url(UPSTREAM_ENDPOINT_OPENAI_RESPONSES)
    parsed_url = urlparse(target_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
    client = await adapter.http_pool_manager.get_streaming_client(base_url=base_url)
    parser = SSEStreamParser()
    saw_terminal_event = False

    async with client.stream(
        "POST",
        target_url,
        headers=provider_headers,
        content=json.dumps(provider_payload).encode("utf-8"),
        ) as upstream_response:
        if upstream_response.status_code >= 400:
            error_body = await upstream_response.aread()
            try:
                error_payload = json.loads(error_body.decode("utf-8"))
            except Exception:
                error_payload = {
                    "error": {
                        "type": "server_error",
                        "message": error_body.decode("utf-8", errors="replace")
                        or "Upstream Codex request failed",
                    }
                }
            await websocket.send_text(
                json.dumps(
                    _make_websocket_terminal_event(
                        provider_payload,
                        error=error_payload.get("error", error_payload),
                    ),
                    separators=(",", ":"),
                )
            )
            return

        async for chunk in upstream_response.aiter_bytes():
            for event in parser.feed(chunk):
                if event.get("type") in {"response.completed", "response.failed"}:
                    saw_terminal_event = True
                await websocket.send_text(json.dumps(event, separators=(",", ":")))

        for event in parser.flush():
            if event.get("type") in {"response.completed", "response.failed"}:
                saw_terminal_event = True
            await websocket.send_text(json.dumps(event, separators=(",", ":")))

        if not saw_terminal_event:
            await websocket.send_text(
                json.dumps(
                    _make_websocket_terminal_event(
                        provider_payload,
                        error={
                            "type": "server_error",
                            "message": "WebSocket stream ended before response.completed",
                        },
                    ),
                    separators=(",", ":"),
                )
            )


@router.post("/v1/responses", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES], endpoint=UPSTREAM_ENDPOINT_OPENAI_RESPONSES
)
async def codex_responses(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await _codex_responses_handler(request, adapter)


@router.websocket("/v1/responses")
async def codex_responses_websocket(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        adapter = _get_codex_websocket_adapter(websocket)
        local_response_ids: set[str] = set()
        while True:
            raw_message = await websocket.receive_text()
            provider_payload = _parse_websocket_request(raw_message)
            if _is_websocket_warmup_request(provider_payload):
                warmup_event = _make_websocket_terminal_event(provider_payload)
                response_id = warmup_event.get("response", {}).get("id")
                if isinstance(response_id, str) and response_id:
                    local_response_ids.add(response_id)
                await websocket.send_text(
                    json.dumps(warmup_event, separators=(",", ":"))
                )
                continue
            previous_response_id = provider_payload.get("previous_response_id")
            if isinstance(previous_response_id, str) and previous_response_id in local_response_ids:
                provider_payload.pop("previous_response_id", None)
            await _stream_websocket_response(websocket, adapter, provider_payload)
    except WebSocketDisconnect:
        return
    except ValueError as exc:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1008, reason=str(exc))
    except Exception as exc:
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason=str(exc))


@router.post("/responses", response_model=None, include_in_schema=False)
@with_format_chain(
    [FORMAT_OPENAI_RESPONSES], endpoint=UPSTREAM_ENDPOINT_OPENAI_RESPONSES
)
async def codex_responses_legacy(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await _codex_responses_handler(request, adapter)


@router.websocket("/responses")
async def codex_responses_legacy_websocket(websocket: WebSocket) -> None:
    await codex_responses_websocket(websocket)


@router.post("/v1/chat/completions", response_model=None)
@with_format_chain(
    [FORMAT_OPENAI_CHAT, FORMAT_OPENAI_RESPONSES],
    endpoint=UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
)
async def codex_chat_completions(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.get("/v1/models", response_model=None)
async def list_models(
    request: Request,
    auth: ConditionalAuthDep,
    config: CodexConfigDep,
) -> dict[str, Any]:
    """List available Codex models."""
    openai_models = _serialize_codex_models(config)
    codex_models = _serialize_codex_cli_models(config)
    return {"object": "list", "data": openai_models, "models": codex_models}


@router.post("/v1/messages", response_model=None)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_RESPONSES],
    endpoint=UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
)
async def codex_v1_messages(
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)


@router.post("/{session_id}/v1/messages", response_model=None)
@with_format_chain(
    [FORMAT_ANTHROPIC_MESSAGES, FORMAT_OPENAI_RESPONSES],
    endpoint="/{session_id}/v1/messages",
)
async def codex_v1_messages_with_session(
    session_id: str,
    request: Request,
    auth: ConditionalAuthDep,
    adapter: CodexAdapterDep,
) -> StreamingResponse | Response | DeferredStreaming:
    return await handle_codex_request(request, adapter)
