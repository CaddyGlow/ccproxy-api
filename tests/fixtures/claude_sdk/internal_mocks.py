"""Internal mocks for ClaudeSDKService.

These fixtures provide AsyncMock objects for dependency injection testing.
They mock the ClaudeSDKService class directly for use with app.dependency_overrides.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from claude_code_sdk import (
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)

from ccproxy.core.errors import ClaudeProxyError


@pytest.fixture
def mock_internal_claude_sdk_service() -> AsyncMock:
    """Create a mock Claude SDK service for internal dependency injection."""
    mock_service = AsyncMock()
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    async def mock_create_completion(*args: Any, **kwargs: Any) -> dict[str, Any]:
        model = kwargs.get("model", "")
        if model not in SUPPORTED_MODELS:
            raise ClaudeProxyError(
                message=f"Unsupported model: {model}",
                error_type="invalid_request_error",
                status_code=400,
            )
        return {
            "id": "msg_01234567890",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        }

    mock_service.create_completion = mock_create_completion
    mock_service.list_models.return_value = [
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
    ]
    mock_service.validate_health.return_value = True
    return mock_service


@pytest.fixture
def mock_internal_claude_sdk_service_unavailable() -> AsyncMock:
    """Create a mock Claude SDK service that simulates service unavailability."""
    mock_service = AsyncMock()

    async def mock_create_completion_error(*args: Any, **kwargs: Any) -> None:
        raise ClaudeProxyError(
            message="Claude SDK service is currently unavailable",
            error_type="service_unavailable",
            status_code=503,
        )

    mock_service.create_completion = mock_create_completion_error
    mock_service.validate_health.return_value = False
    return mock_service


@pytest.fixture
def mock_internal_claude_sdk_service_streaming() -> AsyncMock:
    """Create a mock Claude SDK service for streaming response testing."""

    async def mock_streaming_response() -> AsyncGenerator[dict[str, Any], None]:
        """Mock streaming response generator."""
        events: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " world!"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
        for event in events:
            yield event

    mock_service = AsyncMock()

    async def mock_create_completion(*args: Any, **kwargs: Any) -> Any:
        SUPPORTED_MODELS = [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "claude-3-haiku-20240307",
        ]
        model = kwargs.get("model", "")
        if model not in SUPPORTED_MODELS:
            raise ClaudeProxyError(
                message=f"Unsupported model: {model}",
                error_type="invalid_request_error",
                status_code=400,
            )
        if kwargs.get("stream", False):
            return mock_streaming_response()
        else:
            return {
                "id": "msg_01234567890",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hello! How can I help you?"}],
                "model": model,
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 8},
            }

    mock_service.create_completion = mock_create_completion
    mock_service.list_models.return_value = [
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-opus-20240229",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
    ]
    mock_service.validate_health.return_value = True
    return mock_service


@pytest.fixture
def mock_claude_sdk_client_streaming() -> AsyncMock:
    """Create a mock Claude SDK client for streaming response testing."""
    mock_client = AsyncMock()

    async def mock_query_completion(
        *args: Any, **kwargs: Any
    ) -> AsyncGenerator[Any, None]:
        yield AssistantMessage(
            content=[TextBlock(text="Hello")],
            session_id="test_session",
            stop_reason=None,
            stop_sequences=None,
            model="claude-3-5-sonnet-20241022",
            message_id="msg_123",
        )
        yield AssistantMessage(
            content=[TextBlock(text=" world!")],
            session_id="test_session",
            stop_reason=None,
            stop_sequences=None,
            model="claude-3-5-sonnet-20241022",
            message_id="msg_123",
        )
        yield AssistantMessage(
            content=[
                ToolUseBlock(id="tool_123", name="test_tool", input={"arg": "value"})
            ],
            session_id="test_session",
            stop_reason=None,
            stop_sequences=None,
            model="claude-3-5-sonnet-20241022",
            message_id="msg_123",
        )
        # Yield a tool result block
        yield AssistantMessage(
            content=[ToolResultBlock(tool_use_id="tool_123", content="tool output")],
            session_id="test_session",
            stop_reason=None,
            stop_sequences=None,
            model="claude-3-5-sonnet-20241022",
            message_id="msg_123",
        )
        yield ResultMessage(
            session_id="test_session",
            stop_reason="end_turn",
            total_cost_usd=0.001,
            usage={
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        )

    mock_client.query_completion = mock_query_completion
    mock_client.get_last_api_call_time_ms.return_value = 123.45
    return mock_client
