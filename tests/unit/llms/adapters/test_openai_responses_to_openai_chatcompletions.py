import pytest

from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.openai.models import (
    ChatCompletionChunk as OpenAIChatChunk,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)


@pytest.mark.asyncio
async def test_openai_responses_to_openai_chat_adapter_adapt_request_delegates():
    from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    chat_req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=128,
    )
    out = await OpenAIResponsesToOpenAIChatAdapter().adapt_request(chat_req.model_dump())
    resp_req = OpenAIResponseRequest.model_validate(out)
    assert resp_req.model == "gpt-4o"
    assert resp_req.max_output_tokens == 128
    assert resp_req.input and resp_req.input[0]["type"] == "message"


@pytest.mark.asyncio
async def test_openai_responses_to_openai_chat_adapter_adapt_stream_minimal():
    from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    async def gen():
        yield {"type": "response.created", "response": {"model": "gpt-4o"}}
        yield {"type": "response.output_text.delta", "delta": "Hello"}
        yield {
            "type": "response.completed",
            "response": {"usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}},
        }

    adapter = OpenAIResponsesToOpenAIChatAdapter()
    chunks = []
    async for c in adapter.adapt_stream(gen()):
        chunks.append(OpenAIChatChunk.model_validate(c))

    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"

