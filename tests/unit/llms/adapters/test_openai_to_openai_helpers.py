import pytest

from ccproxy.llms.openai import models as openai_models


@pytest.mark.asyncio
async def test_convert__openai_chat_to_openai_response__response_basic() -> None:
    from ccproxy.llms.adapters.openai_to_openai import helpers as h

    chat_resp = openai_models.ChatCompletionResponse(
        id="chatcmpl_1",
        object="chat.completion",
        created=0,
        model="gpt-4o",
        usage=None,
        choices=[
            openai_models.Choice(
                index=0,
                message=openai_models.ResponseMessage(role="assistant", content="Hi"),
                finish_reason="stop",
            )
        ],
    )

    out = await h.convert__openai_chat_to_openai_response__response(chat_resp)
    assert out.object == "response"
    assert out.model == "gpt-4o"
    assert out.output and out.output[0].type == "message"
    # content aggregated into output_text
    assert out.output[0].content[0].type == "output_text"
    assert out.output[0].content[0].text == "Hi"


@pytest.mark.asyncio
async def test_convert__openai_response_to_openai_chat__response_basic() -> None:
    from ccproxy.llms.adapters.openai_to_openai import helpers as h

    resp = openai_models.ResponseObject(
        id="res_x",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-4o",
        output=[
            openai_models.MessageOutput(
                type="message",
                id="m1",
                status="completed",
                role="assistant",
                content=[
                    openai_models.OutputTextContent(type="output_text", text="Hello")
                ],
            )
        ],
        parallel_tool_calls=False,
        usage=openai_models.ResponseUsage(
            input_tokens=1,
            input_tokens_details=openai_models.InputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=openai_models.OutputTokensDetails(reasoning_tokens=0),
            total_tokens=3,
        ),
    )

    out = h.convert__openai_response_to_openai_chat__response(resp)
    assert out.object == "chat.completion"
    assert out.choices and out.choices[0].message.content == "Hello"


@pytest.mark.asyncio
async def test_convert__openai_response_to_openai_chat__stream_minimal() -> None:
    from ccproxy.llms.adapters.openai_to_openai import helpers as h

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

    chunks = []
    async for c in h.convert__openai_response_to_openai_chat__stream(gen()):
        chunks.append(c)

    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[-1].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_convert__openai_response_to_openaichat__request_maps_core_fields() -> (
    None
):
    from ccproxy.llms.adapters.openai_to_openai import helpers as h

    req = openai_models.ResponseRequest(
        model="gpt-4o",
        input=[
            openai_models.InputMessage(
                role="user",
                content=[
                    openai_models.InputTextContent(type="input_text", text="Hello")
                ],
            )
        ],
        max_output_tokens=256,
        stream=True,
        temperature=0.7,
        top_p=0.9,
        tools=[{"type": "function", "function": {"name": "calc"}}],
        tool_choice="auto",
        parallel_tool_calls=True,
    )

    out = await h.convert__openai_response_to_openaichat__request(req)
    assert out.model == "gpt-4o"
    assert out.messages[-1].role == "user"
    assert out.max_completion_tokens == 256
    assert out.stream is True
    assert out.temperature == 0.7
    assert out.top_p == 0.9
    assert out.tools and out.tool_choice == "auto"
    assert out.parallel_tool_calls is True
