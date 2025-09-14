import pytest

from ccproxy.llms.anthropic.models import (
    ContentBlockDeltaEvent as AnthropicContentBlockDeltaEvent,
)
from ccproxy.llms.anthropic.models import (
    ContentBlockStartEvent as AnthropicContentBlockStartEvent,
)
from ccproxy.llms.anthropic.models import (
    ContentBlockStopEvent as AnthropicContentBlockStopEvent,
)
from ccproxy.llms.anthropic.models import (
    MessageDelta as AnthropicMessageDelta,
)
from ccproxy.llms.anthropic.models import (
    MessageDeltaEvent as AnthropicMessageDeltaEvent,
)
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    MessageStartEvent as AnthropicMessageStartEvent,
)
from ccproxy.llms.anthropic.models import (
    MessageStopEvent as AnthropicMessageStopEvent,
)
from ccproxy.llms.anthropic.models import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.anthropic.models import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_adapt_response_delegates():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    anth = AnthropicMessageResponse(
        id="msg_2",
        role="assistant",
        model="claude",
        content=[AnthropicTextBlock(type="text", text="Hello")],
        stop_reason="end_turn",
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=1, output_tokens=2),
    )
    out = await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_response(
        anth.model_dump()
    )
    OpenAIResponseObject.model_validate(out)


@pytest.mark.asyncio
async def test_openai_responses_request_to_anthropic_messages_adapt_stream_minimal():
    from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
        OpenAIResponsesRequestToAnthropicMessagesAdapter,
    )

    msg = AnthropicMessageResponse(
        id="mstream2",
        role="assistant",
        model="claude",
        content=[],
        stop_reason=None,
        stop_sequence=None,
        usage=AnthropicUsage(input_tokens=0, output_tokens=0),
    )
    events = [
        AnthropicMessageStartEvent(type="message_start", message=msg).model_dump(),
        AnthropicContentBlockStartEvent(
            type="content_block_start",
            index=0,
            content_block=AnthropicTextBlock(type="text", text=""),
        ).model_dump(),
        AnthropicContentBlockDeltaEvent(
            type="content_block_delta",
            index=0,
            delta=AnthropicTextBlock(type="text", text="Hi"),
        ).model_dump(),
        AnthropicContentBlockStopEvent(type="content_block_stop", index=0).model_dump(),
        AnthropicMessageDeltaEvent(
            type="message_delta",
            delta=AnthropicMessageDelta(stop_reason="end_turn"),
            usage=AnthropicUsage(input_tokens=1, output_tokens=2),
        ).model_dump(),
        AnthropicMessageStopEvent(type="message_stop").model_dump(),
    ]

    async def gen():
        for e in events:
            yield e

    adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
    res_events = []
    async for ev in adapter.adapt_stream(gen()):
        res_events.append(ev)

    assert any(
        e.get("type") == "response.output_text.delta" and e.get("delta") == "Hi"
        for e in res_events
    )
    assert any(e.get("type") == "response.completed" for e in res_events)
