from datetime import datetime

import pytest

from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.plugins.request_tracer.config import RequestTracerConfig
from ccproxy.plugins.request_tracer.hook import RequestTracerHook


@pytest.mark.asyncio
async def test_streaming_response_saved_with_chunk_logging_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_id = "req-test"
    chunk = b"data: hello\n\n"

    config = RequestTracerConfig(
        log_streaming_chunks=False,
    )
    hook = RequestTracerHook(config=config)

    recorded: dict[str, object] = {}

    async def fake_write_streaming_response_file(
        self: RequestTracerHook,
        req_id: str,
        chunks: list[bytes],
        metadata: dict[str, object],
    ) -> None:
        recorded["request_id"] = req_id
        recorded["chunks"] = chunks
        recorded["metadata"] = metadata

    monkeypatch.setattr(
        RequestTracerHook,
        "_write_streaming_response_file",
        fake_write_streaming_response_file,
        raising=True,
    )

    await hook(
        HookContext(
            event=HookEvent.PROVIDER_STREAM_START,
            timestamp=datetime.now(),
            provider="example",
            data={
                "request_id": request_id,
                "url": "https://example.com",
                "method": "GET",
                "buffered_mode": False,
            },
            metadata={"request_id": request_id},
        )
    )

    await hook(
        HookContext(
            event=HookEvent.PROVIDER_STREAM_CHUNK,
            timestamp=datetime.now(),
            provider="example",
            data={
                "request_id": request_id,
                "chunk": chunk,
                "chunk_number": 1,
                "chunk_size": len(chunk),
            },
            metadata={"request_id": request_id},
        )
    )

    await hook(
        HookContext(
            event=HookEvent.PROVIDER_STREAM_END,
            timestamp=datetime.now(),
            provider="example",
            data={
                "request_id": request_id,
                "total_chunks": 1,
                "total_bytes": len(chunk),
                "upstream_stream_text": "event: raw\ndata: raw\n\n",
            },
            metadata={"request_id": request_id},
        )
    )

    assert recorded["request_id"] == request_id
    assert recorded["chunks"] == [chunk]
    metadata = recorded["metadata"]
    assert isinstance(metadata, dict)
    assert metadata["total_chunks"] == 1
    assert metadata["total_bytes"] == len(chunk)
    assert metadata["upstream_stream_text"] == "event: raw\ndata: raw\n\n"
