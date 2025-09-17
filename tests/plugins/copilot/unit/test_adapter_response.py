"""Tests for Copilot adapter response normalization."""

import json
from unittest.mock import MagicMock

import httpx
import pytest

from ccproxy.llms.models.openai import ResponseObject
from ccproxy.plugins.copilot.adapter import CopilotAdapter
from ccproxy.plugins.copilot.config import CopilotConfig


@pytest.mark.asyncio
async def test_process_provider_response_adds_missing_created_timestamp() -> None:
    """Ensure chat completions responses always include the required field."""

    adapter = CopilotAdapter(
        oauth_provider=MagicMock(),
        config=CopilotConfig(),
        auth_manager=object(),
        detection_service=object(),
        http_pool_manager=object(),
    )

    provider_payload = {
        "id": "chatcmpl-123",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "model": "gpt-4o",
    }

    provider_response = httpx.Response(
        status_code=200,
        json=provider_payload,
        headers={"Content-Type": "application/json"},
    )

    result = await adapter.process_provider_response(
        provider_response, "/chat/completions"
    )

    body = json.loads(result.body)

    assert "created" in body
    assert isinstance(body["created"], int)


@pytest.mark.asyncio
async def test_process_provider_response_normalizes_response_object() -> None:
    """Ensure Response API payloads are normalized to OpenAI schema."""

    adapter = CopilotAdapter(
        oauth_provider=MagicMock(),
        config=CopilotConfig(),
        auth_manager=object(),
        detection_service=object(),
        http_pool_manager=object(),
    )

    provider_payload = {
        "id": "msg_test",
        "model": "claude-sonnet",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi there"},
                ],
            }
        ],
        "stop_reason": "end_turn",
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
        },
    }

    provider_response = httpx.Response(
        status_code=200,
        json=provider_payload,
        headers={"Content-Type": "application/json"},
    )

    result = await adapter.process_provider_response(provider_response, "/responses")

    body = json.loads(result.body)

    # Validate against canonical model and ensure key fields are present
    normalized = ResponseObject.model_validate(body)
    assert normalized.object == "response"
    assert normalized.status == "completed"
    assert normalized.output[0].content[0].type == "output_text"
