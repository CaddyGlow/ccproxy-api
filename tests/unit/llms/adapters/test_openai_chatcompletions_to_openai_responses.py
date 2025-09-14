import pytest

from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)
from ccproxy.llms.openai.models import (
    ResponseUsage as OpenAIResponseUsage,
)
from ccproxy.llms.openai.models import (
    InputTokensDetails as OpenAIInputTokensDetails,
)
from ccproxy.llms.openai.models import (
    OutputTokensDetails as OpenAIOutputTokensDetails,
)


@pytest.mark.asyncio
async def test_openai_chat_to_openai_responses_adapter_adapt_response_delegates():
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    resp = OpenAIResponseObject(
        id="res_x",
        object="response",
        created_at=0,
        status="completed",
        model="gpt-4o",
        output=[
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Hi"}],
            }
        ],
        parallel_tool_calls=False,
        usage=OpenAIResponseUsage(
            input_tokens=1,
            input_tokens_details=OpenAIInputTokensDetails(cached_tokens=0),
            output_tokens=2,
            output_tokens_details=OpenAIOutputTokensDetails(reasoning_tokens=0),
            total_tokens=3,
        ),
    )

    out = await OpenAIChatToOpenAIResponsesAdapter().adapt_response(resp.model_dump())
    # Adapter returns a Chat response, validate minimal fields by presence
    assert isinstance(out, dict) and out.get("object") == "chat.completion"

