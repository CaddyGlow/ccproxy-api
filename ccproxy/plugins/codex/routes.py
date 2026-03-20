"""Codex plugin routes."""

import contextlib
import json
from collections import deque
from pathlib import Path
from time import time
from typing import TYPE_CHECKING, Annotated, Any, cast
from urllib.parse import urlparse
from uuid import uuid4

import anyio
from fastapi import APIRouter, Depends, Request, WebSocket, WebSocketDisconnect
from starlette.responses import Response, StreamingResponse
from starlette.websockets import WebSocketState

from ccproxy.api.decorators import with_format_chain
from ccproxy.api.dependencies import (
    get_plugin_adapter,
    get_provider_config_dependency,
)
from ccproxy.auth.dependencies import ConditionalAuthDep
from ccproxy.config.settings import Settings
from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
    UPSTREAM_ENDPOINT_ANTHROPIC_MESSAGES,
    UPSTREAM_ENDPOINT_OPENAI_CHAT_COMPLETIONS,
    UPSTREAM_ENDPOINT_OPENAI_RESPONSES,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins import PluginRegistry, ProviderPluginRuntime
from ccproxy.streaming import DeferredStreaming
from ccproxy.streaming.sse_parser import SSEStreamParser

from .config import CodexSettings


if TYPE_CHECKING:
    from .adapter import CodexAdapter

logger = get_plugin_logger()

_MAX_LOCAL_RESPONSE_IDS = 256

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


def _get_codex_websocket_adapter(websocket: WebSocket) -> "CodexAdapter":
    if not hasattr(websocket.app.state, "plugin_registry"):
        raise RuntimeError("Plugin registry not initialized")

    registry: PluginRegistry = websocket.app.state.plugin_registry
    runtime = registry.get_runtime("codex")

    if not runtime or not isinstance(runtime, ProviderPluginRuntime):
        raise RuntimeError("Codex plugin not initialized")

    if not runtime.adapter:
        raise RuntimeError("Codex adapter not available")

    return cast("CodexAdapter", runtime.adapter)


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
    response_payload: dict[str, Any] = {
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


async def _authenticate_websocket(websocket: WebSocket) -> None:
    """Enforce bearer auth on WebSocket connections when auth is configured.

    Mirrors the ConditionalAuthDep logic: if security.auth_token is set,
    the client must provide a matching Authorization header. Closes the
    connection with 1008 (Policy Violation) on failure.
    """
    container = getattr(websocket.app.state, "service_container", None)
    settings: Settings | None = None
    if container is not None:
        with contextlib.suppress(ValueError):
            settings = container.get_service(Settings)
    if settings is None:
        with contextlib.suppress(Exception):
            settings = Settings()

    if settings is None or not settings.security.auth_token:
        return

    expected = settings.security.auth_token.get_secret_value()
    auth_header = websocket.headers.get("authorization", "")
    scheme, _, credentials = auth_header.partition(" ")
    if scheme.lower() == "bearer":
        credentials = credentials.strip()
        token = credentials.split()[0] if credentials else ""
    else:
        token = ""

    if token != expected:
        await websocket.close(code=1008, reason="Authentication required")
        raise WebSocketDisconnect(code=1008)


async def _sanitize_websocket_payload(
    adapter: "CodexAdapter", provider_payload: dict[str, Any], headers: dict[str, str]
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the same request normalization used by HTTP routes on a WS payload."""
    body_bytes = json.dumps(provider_payload).encode("utf-8")
    prepared_body, prepared_headers = await adapter.prepare_provider_request(
        body_bytes, headers, UPSTREAM_ENDPOINT_OPENAI_RESPONSES
    )
    sanitized_payload = json.loads(prepared_body.decode("utf-8"))
    return sanitized_payload, prepared_headers


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


async def _load_codex_cli_models_cache() -> list[dict[str, Any]]:
    cache_path = anyio.Path(Path.home() / ".codex" / "models_cache.json")
    if not await cache_path.exists():
        return []

    try:
        content = await cache_path.read_text()
        payload = json.loads(content)
    except Exception:
        return []

    models = payload.get("models")
    if not isinstance(models, list):
        return []

    return [model for model in models if isinstance(model, dict)]


async def _serialize_codex_cli_models(config: CodexSettings) -> list[dict[str, Any]]:
    configured_ids = {
        card.id
        for card in config.models_endpoint
        if isinstance(getattr(card, "id", None), str)
    }
    configured_ids.update(
        {
            card.root
            for card in config.models_endpoint
            if isinstance(getattr(card, "root", None), str) and card.root
        }
    )

    cached_models = await _load_codex_cli_models_cache()
    if cached_models and configured_ids:
        matched = [
            model
            for model in cached_models
            if model.get("slug") in configured_ids
            or model.get("display_name") in configured_ids
        ]
        if matched:
            return matched

    return _serialize_codex_models(config)


async def _stream_websocket_response(
    websocket: WebSocket,
    adapter: "CodexAdapter",
    provider_payload: dict[str, Any],
) -> None:
    request_headers = _prepare_websocket_headers(websocket)
    provider_payload, provider_headers = await _sanitize_websocket_payload(
        adapter, provider_payload, request_headers
    )
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
    await _authenticate_websocket(websocket)

    try:
        adapter = _get_codex_websocket_adapter(websocket)
        local_response_ids: deque[str] = deque(maxlen=_MAX_LOCAL_RESPONSE_IDS)
        logger.debug("websocket_connected", client=str(websocket.client))
        while True:
            raw_message = await websocket.receive_text()
            provider_payload = _parse_websocket_request(raw_message)
            if _is_websocket_warmup_request(provider_payload):
                warmup_event = _make_websocket_terminal_event(provider_payload)
                response_id = warmup_event.get("response", {}).get("id")
                if isinstance(response_id, str) and response_id:
                    local_response_ids.append(response_id)
                await websocket.send_text(
                    json.dumps(warmup_event, separators=(",", ":"))
                )
                logger.debug("websocket_warmup_handled", response_id=response_id)
                continue
            previous_response_id = provider_payload.get("previous_response_id")
            if (
                isinstance(previous_response_id, str)
                and previous_response_id in local_response_ids
            ):
                provider_payload.pop("previous_response_id", None)
            logger.debug(
                "websocket_streaming_request", model=provider_payload.get("model")
            )
            await _stream_websocket_response(websocket, adapter, provider_payload)
    except WebSocketDisconnect:
        logger.debug("websocket_disconnected", client=str(websocket.client))
        return
    except ValueError as exc:
        logger.warning("websocket_value_error", error=str(exc))
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1008, reason=str(exc))
    except Exception:
        logger.warning("websocket_unexpected_error", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal server error")


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
    codex_models = await _serialize_codex_cli_models(config)
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
