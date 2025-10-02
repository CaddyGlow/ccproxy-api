from collections.abc import AsyncGenerator
from typing import Any

import pytest

from ccproxy.llms.formatters import openai_to_openai as formatter_module


pytestmark = pytest.mark.asyncio


async def _dummy_stream() -> AsyncGenerator[None, None]:
    yield None


async def test_convert_responses_stream_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_stream()
    captured: dict[str, Any] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream

            async def generator():
                yield "chunk"

            return generator()

    monkeypatch.setattr(
        formatter_module,
        "OpenAIResponsesToChatStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__openai_responses_to_openai_chat__stream(
        stream_instance  # type: ignore[arg-type]
    )

    chunk = await anext(result)
    assert chunk == "chunk"  # type: ignore[comparison-overlap]
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance


async def test_convert_chat_stream_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_stream()
    captured: dict[str, Any] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream

            async def generator():
                yield {"type": "chunk"}

            return generator()

    monkeypatch.setattr(
        formatter_module,
        "OpenAIChatToResponsesStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__openai_chat_to_openai_responses__stream(
        stream_instance  # type: ignore[arg-type]
    )

    chunk = await anext(result)
    assert chunk == {"type": "chunk"}  # type: ignore[comparison-overlap]
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance
