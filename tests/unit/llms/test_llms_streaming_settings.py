import pytest

from ccproxy.llms.streaming.processors import OpenAIStreamProcessor


async def _gen_chunks():
    yield {"type": "message_start"}
    # Thinking block
    yield {"type": "content_block_start", "content_block": {"type": "thinking"}}
    yield {
        "type": "content_block_delta",
        "delta": {"type": "thinking_delta", "thinking": "secret"},
    }
    yield {
        "type": "content_block_delta",
        "delta": {"type": "signature_delta", "signature": "sig"},
    }
    yield {"type": "content_block_stop"}
    # Visible text
    yield {"type": "content_block_start", "content_block": {"type": "text", "text": ""}}
    yield {
        "type": "content_block_delta",
        "delta": {"type": "text_delta", "text": "hello"},
    }
    yield {"type": "content_block_stop"}
    yield {"type": "message_delta", "usage": {"input_tokens": 1, "output_tokens": 1}}
    yield {"type": "message_stop"}


@pytest.mark.asyncio
async def test_llm_openai_thinking_xml_env_disables_thinking_serialization(monkeypatch):
    monkeypatch.setenv("LLM__OPENAI_THINKING_XML", "false")

    proc = OpenAIStreamProcessor(output_format="dict")
    out = []
    async for chunk in proc.process_stream(_gen_chunks()):
        assert isinstance(chunk, dict)
        out.append(chunk)

    # Ensure no thinking XML appears in any content delta
    for c in out:
        if c.get("choices"):
            delta = c["choices"][0].get("delta") or {}
            if isinstance(delta, dict) and "content" in delta:
                assert "<thinking" not in delta["content"]
