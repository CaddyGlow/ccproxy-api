import pytest

from ccproxy.llms.formatters import openai_to_anthropic as formatter_module


pytestmark = pytest.mark.asyncio


async def _dummy_responses_stream() -> None:
    yield None


async def _dummy_chat_stream() -> None:
    yield None


async def test_responses_stream_wrapper_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_responses_stream()
    captured: dict[str, object] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        async def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream
            yield "anthropic-event"

    monkeypatch.setattr(
        formatter_module,
        "OpenAIResponsesToAnthropicStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__openai_responses_to_anthropic_messages__stream(
        stream_instance
    )

    event = await anext(result)
    assert event == "anthropic-event"
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance


async def test_chat_stream_wrapper_uses_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream_instance = _dummy_chat_stream()
    captured: dict[str, object] = {}

    class DummyAdapter:
        def __init__(self) -> None:
            captured["init"] = True

        def run(self, stream):  # type: ignore[no-untyped-def]
            captured["stream"] = stream

            async def generator():
                yield {"type": "anthropic-chunk"}

            return generator()

    monkeypatch.setattr(
        formatter_module,
        "OpenAIChatToAnthropicStreamAdapter",
        DummyAdapter,
    )

    result = formatter_module.convert__openai_chat_to_anthropic_messages__stream(
        stream_instance
    )

    chunk = await anext(result)
    assert chunk == {"type": "anthropic-chunk"}
    await result.aclose()
    await stream_instance.aclose()

    assert captured["init"] is True
    assert captured["stream"] is stream_instance
