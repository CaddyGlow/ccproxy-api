"""Internal mocks for ClaudeSDKService.

These fixtures provide AsyncMock objects for dependency injection testing.
They mock the ClaudeSDKService class directly for use with app.dependency_overrides.
"""

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def mock_internal_claude_sdk_service() -> AsyncMock:
    """Create a mock Claude SDK service for internal dependency injection.

    This fixture creates an AsyncMock that replaces ClaudeSDKService in the
    dependency injection container. Use this for testing API endpoints that
    depend on Claude SDK without making actual HTTP calls.

    Mocking Strategy: Internal service mocking via AsyncMock
    Use Case: Testing API endpoints with dependency injection
    HTTP Calls: None (internal mocking only)
    """
    from ccproxy.core.errors import ClaudeProxyError

    mock_service = AsyncMock()

    # List of supported models for validation
    SUPPORTED_MODELS = [
        "claude-3-5-sonnet-20241022",
        "claude-3-5-sonnet-20240620",
        "claude-3-opus-20240229",
        "claude-3-haiku-20240307",
    ]

    # Mock create_completion with model validation
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

    # Mock the list_models method
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

    # Mock the validate_health method
    mock_service.validate_health.return_value = True

    return mock_service


@pytest.fixture
def mock_internal_claude_sdk_service_unavailable() -> AsyncMock:
    """Create a mock Claude SDK service that simulates service unavailability.

    This fixture creates an AsyncMock that raises errors to simulate when
    the Claude SDK service is unavailable or misconfigured.

    Mocking Strategy: Internal service mocking via AsyncMock
    Use Case: Testing error handling when Claude SDK is unavailable
    HTTP Calls: None (internal mocking only)
    """
    from ccproxy.core.errors import ClaudeProxyError

    mock_service = AsyncMock()

    # Mock create_completion to raise an error
    async def mock_create_completion_error(*args: Any, **kwargs: Any) -> None:
        raise ClaudeProxyError(
            message="Claude SDK service is currently unavailable",
            error_type="service_unavailable",
            status_code=503,
        )

    mock_service.create_completion = mock_create_completion_error

    # Mock the validate_health method to return False
    mock_service.validate_health.return_value = False

    return mock_service


@pytest.fixture
def mock_internal_claude_sdk_service_streaming() -> AsyncMock:
    """Create a mock Claude SDK service for streaming response testing.

    This fixture creates an AsyncMock that supports both regular and streaming
    responses for comprehensive testing of the streaming functionality.

    Mocking Strategy: Internal service mocking via AsyncMock
    Use Case: Testing streaming API endpoints with dependency injection
    HTTP Calls: None (internal mocking only)
    """

    async def mock_streaming_response() -> AsyncGenerator[dict[str, Any], None]:
        """Mock streaming response generator."""
        events = [
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
            yield event  # type: ignore[misc]

    mock_service = AsyncMock()

    # Mock create_completion as an async function with model validation
    async def mock_create_completion(*args: Any, **kwargs: Any) -> Any:
        from ccproxy.core.errors import ClaudeProxyError

        # List of supported models for validation
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

    # Mock the list_models method
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

    # Mock the validate_health method
    mock_service.validate_health.return_value = True

    return mock_service
