from __future__ import annotations

import time
from collections.abc import MutableMapping
from typing import Any

import anyio
import pytest

from ccproxy.api.middleware.streaming_hooks import StreamingResponseWithHooks
from ccproxy.core.logging import get_logger
from ccproxy.core.plugins.hooks import HookManager
from ccproxy.core.plugins.hooks.base import Hook, HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.core.plugins.hooks.registry import HookRegistry
from ccproxy.core.request_context import RequestContext


class _StreamingCollectorHook(Hook):
    name = "streaming_collector"
    events = [HookEvent.HTTP_RESPONSE, HookEvent.REQUEST_COMPLETED]
    priority = 500

    def __init__(self, results: dict[str, set[HookEvent]]) -> None:
        self._results = results

    async def __call__(self, context: HookContext) -> None:
        request_id = context.data.get("request_id")
        if not request_id:
            return
        self._results.setdefault(request_id, set()).add(context.event)


class _DummyStreamResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks
        self.status_code = 200
        self.headers = {"content-type": "text/event-stream"}
        self.media_type = "text/event-stream"

    async def aiter_bytes(self) -> Any:
        for chunk in self._chunks:
            await anyio.sleep(0)
            yield chunk

    async def aread(self) -> bytes:
        return b"".join(self._chunks)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_streaming_hooks_stress() -> None:
    registry = HookRegistry()
    results: dict[str, set[HookEvent]] = {}
    collector = _StreamingCollectorHook(results)
    registry.register(collector)

    hook_manager = HookManager(registry)

    total_streams = 40
    chunks_per_stream = 5

    async def run_stream(index: int) -> None:
        request_id = f"stream-stress-{index}"
        request_context = RequestContext(
            request_id=request_id,
            start_time=time.perf_counter(),
            logger=get_logger(__name__).bind(test="stream-stress"),
            metadata={"service_type": "stream-stress"},
        )

        stream_chunks = [
            f"chunk-{index}-{idx}\n".encode() for idx in range(chunks_per_stream)
        ]
        response = StreamingResponseWithHooks(
            content=_DummyStreamResponse(stream_chunks).aiter_bytes(),
            hook_manager=hook_manager,
            request_id=request_id,
            request_data={
                "method": "GET",
                "url": f"https://example.test/stream/{index}",
                "headers": {"accept": "text/event-stream"},
            },
            request_metadata=request_context.metadata,
            start_time=time.perf_counter(),
            status_code=200,
        )

        scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "GET",
            "path": f"/stream/{index}",
            "headers": [],
        }

        receive_called = False

        async def receive() -> dict[str, Any]:  # noqa: ANN401
            nonlocal receive_called
            if receive_called:
                await anyio.sleep(0)
                return {"type": "http.disconnect"}
            receive_called = True
            return {"type": "http.request", "body": b"", "more_body": False}

        async def send(message: MutableMapping[str, Any]) -> None:
            # Consume messages without storing to keep test lightweight
            if message.get("type") == "http.response.body" and not message.get(
                "more_body", False
            ):
                await anyio.sleep(0)

        await response(scope, receive, send)

    async with anyio.create_task_group() as tg:
        for stream_index in range(total_streams):
            tg.start_soon(run_stream, stream_index)

    # Allow background hook tasks to flush
    await anyio.sleep(0.05)

    assert len(results) == total_streams
    for recorded_events in results.values():
        assert HookEvent.HTTP_RESPONSE in recorded_events
        assert HookEvent.REQUEST_COMPLETED in recorded_events
