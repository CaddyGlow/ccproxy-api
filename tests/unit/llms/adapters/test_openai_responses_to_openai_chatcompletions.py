import pytest

from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)


@pytest.mark.asyncio
async def test_openai_responses_to_openai_chat_adapter_adapt_request_delegates() -> (
    None
):
    from ccproxy.llms.adapters.openai_to_openai.responses_to_chat import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    chat_req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        max_completion_tokens=128,
    )
    out = await OpenAIResponsesToOpenAIChatAdapter().adapt_request(chat_req)
    resp_req = OpenAIResponseRequest.model_validate(out.model_dump())
    assert resp_req.model == "gpt-4o"
    assert resp_req.max_output_tokens == 128
    assert resp_req.input and resp_req.input[0]["type"] == "message"


@pytest.mark.asyncio
async def test_openai_responses_to_openai_chat_adapter_adapt_stream_minimal() -> None:
    from ccproxy.llms.adapters.openai_to_openai.responses_to_chat import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    async def gen():
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=1,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                ),
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=2,
                item_id="item_1",
                output_index=0,
                content_index=0,
                delta="Hello",
            )
        )
        yield openai_models.AnyStreamEvent(
            root=openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=3,
                response=openai_models.ResponseObject(
                    id="resp_1",
                    object="response",
                    created_at=0,
                    status="completed",
                    model="gpt-4o",
                    output=[],
                    parallel_tool_calls=False,
                    usage=openai_models.ResponseUsage(
                        input_tokens=1,
                        output_tokens=2,
                        total_tokens=3,
                        input_tokens_details=openai_models.InputTokensDetails(
                            cached_tokens=0
                        ),
                        output_tokens_details=openai_models.OutputTokensDetails(
                            reasoning_tokens=0
                        ),
                    ),
                ),
            )
        )

    adapter = OpenAIResponsesToOpenAIChatAdapter()
    chunks = []
    async for c in adapter.adapt_stream(gen()):
        chunks.append(c)

    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"
