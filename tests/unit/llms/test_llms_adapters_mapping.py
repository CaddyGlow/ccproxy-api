import pytest

from ccproxy.llms.anthropic.models import (
    CreateMessageRequest as AnthropicCreateMessageRequest,
    MessageResponse as AnthropicMessageResponse,
    TextBlock as AnthropicTextBlock,
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
    ChatCompletionResponse as OpenAIChatResponse,
    ResponseObject as OpenAIResponseObject,
    ResponseRequest as OpenAIResponseRequest,
    MessageOutput as OpenAIResponseMessage,
    OutputTextContent as OpenAIOutputTextContent,
    ResponseUsage as OpenAIResponseUsage,
    InputTokensDetails as OpenAIInputTokensDetails,
    OutputTokensDetails as OpenAIOutputTokensDetails,
)


import pytest_asyncio


@pytest.mark.asyncio
async def test_openai_chat_to_anthropic_request_basic():
    from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
        OpenAIChatToAnthropicMessagesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        max_completion_tokens=100,
    )

    adapter = OpenAIChatToAnthropicMessagesAdapter()
    out = await adapter.adapt_request(req.model_dump())

    # Validates as Anthropic create message request
    anth_req = AnthropicCreateMessageRequest.model_validate(out)

    assert anth_req.model == "gpt-4o"
    assert anth_req.max_tokens == 100
    assert anth_req.system == "you are helpful"
    assert len(anth_req.messages) == 1
    assert anth_req.messages[0].role == "user"
    assert anth_req.messages[0].content == "Hello"


@pytest.mark.asyncio
async def test_anthropic_response_to_openai_chat_response():
    from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
        AnthropicMessagesToOpenAIChatAdapter,
    )

    # Minimal anthropic response with 1 text block
    anthropic_resp = AnthropicMessageResponse(
        id="msg_123",
        role="assistant",
        model="claude-sonnet",
        content=[AnthropicTextBlock(type="text", text="Hi there")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=10, output_tokens=5),
    )

    adapter = AnthropicMessagesToOpenAIChatAdapter()
    out = await adapter.adapt_response(anthropic_resp.model_dump())

    openai_resp = OpenAIChatResponse.model_validate(out)
    assert openai_resp.id == "msg_123"
    assert openai_resp.model == "claude-sonnet"
    assert openai_resp.choices[0].message.content == "Hi there"
    assert openai_resp.choices[0].finish_reason == "stop"
    assert openai_resp.usage.prompt_tokens == 10
    assert openai_resp.usage.completion_tokens == 5
    assert openai_resp.usage.total_tokens == 15


@pytest.mark.asyncio
async def test_openai_chat_to_openai_response_request():
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hello"},
        ],
        max_completion_tokens=256,
    )
    adapter = OpenAIChatToOpenAIResponsesAdapter()
    out = await adapter.adapt_request(req.model_dump())
    openai_resp_req = OpenAIResponseRequest.model_validate(out)
    assert openai_resp_req.model == "gpt-4o"
    # We map last user message into a single input message
    assert openai_resp_req.input and openai_resp_req.input[0]["type"] == "message"
    assert openai_resp_req.max_output_tokens == 256


@pytest.mark.asyncio
async def test_openai_response_to_openai_chat_response():
    from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
        OpenAIResponsesToOpenAIChatAdapter,
    )

    resp = OpenAIResponseObject(
        id="res_1",
        object="response",
        created_at=0,
        model="gpt-4o",
        output=[
            OpenAIResponseMessage(
                type="message",
                id="m1",
                status="completed",
                role="assistant",
                content=[OpenAIOutputTextContent(type="output_text", text="Hello!")],
            )
        ],
        status="completed",
        parallel_tool_calls=False,
        usage=OpenAIResponseUsage(
            input_tokens=7,
            input_tokens_details=OpenAIInputTokensDetails(cached_tokens=0),
            output_tokens=9,
            output_tokens_details=OpenAIOutputTokensDetails(reasoning_tokens=0),
            total_tokens=16,
        ),
    )

    adapter = OpenAIResponsesToOpenAIChatAdapter()
    out = await adapter.adapt_response(resp.model_dump())
    chat = OpenAIChatResponse.model_validate(out)
    assert chat.id == "res_1"
    assert chat.model == "gpt-4o"
    assert chat.choices[0].message.content == "Hello!"
    assert chat.usage.total_tokens == 16
