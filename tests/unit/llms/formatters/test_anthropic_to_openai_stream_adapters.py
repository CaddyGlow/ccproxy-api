import pytest

from ccproxy.llms.formatters import anthropic_to_openai as formatter_module


pytestmark = pytest.mark.asyncio


async def _dummy_stream() -> None:
    yield None


async def test_responses_stream_wrapper_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_stream()
    captured: dict[str, object] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        async def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream
            yield "event"

    monkeypatch.setattr(
        formatter_module,
        "AnthropicToOpenAIResponsesStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__anthropic_message_to_openai_responses__stream(
        stream_instance
    )

    event = await anext(result)
    assert event == "event"
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance


async def test_chat_stream_wrapper_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_stream()
    captured: dict[str, object] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        async def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream
            yield {"type": "chunk"}

    monkeypatch.setattr(
        formatter_module,
        "AnthropicToOpenAIChatStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__anthropic_message_to_openai_chat__stream(
        stream_instance
    )

    chunk = await anext(result)
    assert chunk == {"type": "chunk"}
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance
