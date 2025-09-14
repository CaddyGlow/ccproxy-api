import pytest

from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.openai.models import (
    InputTokensDetails as OpenAIInputTokensDetails,
)
from ccproxy.llms.openai.models import (
    OutputTokensDetails as OpenAIOutputTokensDetails,
)
from ccproxy.llms.openai.models import (
    ResponseObject as OpenAIResponseObject,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)
from ccproxy.llms.openai.models import (
    ResponseUsage as OpenAIResponseUsage,
)


@pytest.mark.asyncio
async def test_openai_chat_to_openai_responses_adapter_adapt_response_delegates() -> (
    None
):
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


@pytest.mark.asyncio
async def test_openai_chat_to_openai_response_basic_request() -> None:
    """Test basic request mapping from ChatCompletion to Response format"""
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
async def test_response_format_text_mapping() -> None:
    """Test response_format text type mapping to text.format field"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    # Test text response format (should map to text.format)
    req_text = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "text"},
    )
    out_text = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(
        req_text.model_dump()
    )
    resp_req_text = OpenAIResponseRequest.model_validate(out_text)

    assert resp_req_text.text is not None
    assert resp_req_text.text.get("format", {}).get("type") == "text"


@pytest.mark.asyncio
async def test_response_format_json_object_mapping() -> None:
    """Test response_format json_object mapping to text.format field"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    req_json = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_object"},
    )
    out_json = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(
        req_json.model_dump()
    )
    resp_req_json = OpenAIResponseRequest.model_validate(out_json)

    assert resp_req_json.text is not None
    assert resp_req_json.text.get("format", {}).get("type") == "json_object"


@pytest.mark.asyncio
async def test_response_format_json_schema_mapping() -> None:
    """Test comprehensive json_schema response_format mapping to text.format field"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    # Test json_schema with full schema structure
    schema = {
        "name": "math_response",
        "schema": {
            "type": "object",
            "properties": {"a": {"type": "number"}},
            "required": ["a"],
            "additionalProperties": False,
        },
        "strict": True,
    }
    req = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        response_format={"type": "json_schema", "json_schema": schema},
    )
    out = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(req.model_dump())
    resp_req = OpenAIResponseRequest.model_validate(out)

    assert resp_req.text is not None
    assert resp_req.text.get("format", {}).get("type") == "json_schema"
    fmt = resp_req.text["format"]
    assert fmt.get("name") == "math_response"
    assert isinstance(fmt.get("schema"), dict)
    assert fmt.get("strict") is True


@pytest.mark.asyncio
async def test_response_format_json_schema_variations() -> None:
    """Test different json_schema variations and their mapping"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    adapter = OpenAIChatToOpenAIResponsesAdapter()

    # Test with strict=False
    schema_non_strict = {
        "name": "flexible_response",
        "schema": {
            "type": "object",
            "properties": {"data": {"type": "string"}},
        },
        "strict": False,
    }
    req_non_strict = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_schema", "json_schema": schema_non_strict},
    )
    out_non_strict = await adapter.adapt_request(req_non_strict.model_dump())
    resp_non_strict = OpenAIResponseRequest.model_validate(out_non_strict)

    assert resp_non_strict.text["format"]["strict"] is False
    assert resp_non_strict.text["format"]["name"] == "flexible_response"

    # Test with minimal schema (no name)
    schema_minimal = {
        "schema": {"type": "object"},
    }
    req_minimal = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "json_schema", "json_schema": schema_minimal},
    )
    out_minimal = await adapter.adapt_request(req_minimal.model_dump())
    resp_minimal = OpenAIResponseRequest.model_validate(out_minimal)

    assert resp_minimal.text["format"]["type"] == "json_schema"
    assert "schema" in resp_minimal.text["format"]


@pytest.mark.asyncio
async def test_response_format_with_other_parameters() -> None:
    """Test response_format mapping alongside other parameters"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    # Test response_format with tools
    req_with_tools = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "calculate something"}],
        response_format={"type": "json_object"},
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "calc",
                    "description": "Calculator",
                    "parameters": {
                        "type": "object",
                        "properties": {"expr": {"type": "string"}},
                    },
                },
            }
        ],
        max_completion_tokens=500,
    )
    out_with_tools = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(
        req_with_tools.model_dump()
    )
    resp_with_tools = OpenAIResponseRequest.model_validate(out_with_tools)

    # Should have both response format and tools
    assert resp_with_tools.text["format"]["type"] == "json_object"
    assert resp_with_tools.tools is not None and len(resp_with_tools.tools) == 1
    assert resp_with_tools.max_output_tokens == 500


@pytest.mark.asyncio
async def test_no_response_format_no_text_field() -> None:
    """Test that no response_format results in no text field"""
    from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
        OpenAIChatToOpenAIResponsesAdapter,
    )

    req_no_format = OpenAIChatRequest(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        # No response_format specified
    )
    out_no_format = await OpenAIChatToOpenAIResponsesAdapter().adapt_request(
        req_no_format.model_dump()
    )
    resp_no_format = OpenAIResponseRequest.model_validate(out_no_format)

    # text field should be None or empty when no response_format is specified
    assert resp_no_format.text is None or resp_no_format.text == {}
