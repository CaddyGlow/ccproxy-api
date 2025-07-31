"""Mock fixtures for Claude SDK client testing.

These fixtures provide mocks for the claude_code_sdk client components
used in unit tests for the ClaudeSDKClient wrapper.
"""
# mypy: disable-error-code="unreachable"

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from claude_code_sdk import (
    AssistantMessage,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    ProcessError,
    ResultMessage,
    SystemMessage,
    TextBlock,
    UserMessage,
)


@pytest.fixture
def mock_sdk_client_instance() -> AsyncMock:
    """Create a mock Claude SDK client instance with standard behavior."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    async def simple_response() -> AsyncGenerator[Any, None]:
        yield AssistantMessage(content=[TextBlock(text="Hello")])

    # receive_response should be a method that returns the generator when called
    mock_client.receive_response = lambda: simple_response()
    return mock_client


@pytest.fixture
def mock_sdk_client_streaming() -> AsyncMock:
    """Create a mock Claude SDK client instance with streaming responses."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    async def streaming_response() -> AsyncGenerator[Any, None]:
        yield UserMessage(content="Hello")
        yield AssistantMessage(content=[TextBlock(text="Hi there!")])
        yield SystemMessage(subtype="test", data={"message": "System message"})
        yield ResultMessage(
            subtype="success",
            duration_ms=1000,
            duration_api_ms=800,
            is_error=False,
            num_turns=1,
            session_id="test_session",
            total_cost_usd=0.001,
            usage={"input_tokens": 10, "output_tokens": 5},
        )

    # receive_response should be a method that returns the generator when called
    mock_client.receive_response = lambda: streaming_response()
    return mock_client


@pytest.fixture
def mock_sdk_client_cli_not_found() -> AsyncMock:
    """Create a mock Claude SDK client that raises CLINotFoundError."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock(
        side_effect=CLINotFoundError("Claude CLI not found")
    )
    return mock_client


@pytest.fixture
def mock_sdk_client_cli_connection_error() -> AsyncMock:
    """Create a mock Claude SDK client that raises CLIConnectionError."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock(side_effect=CLIConnectionError("Connection failed"))
    return mock_client


@pytest.fixture
def mock_sdk_client_process_error() -> AsyncMock:
    """Create a mock Claude SDK client that raises ProcessError."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    async def process_error_response() -> AsyncGenerator[Any, None]:
        # Need yield to make it a proper async generator before raising
        if False:  # pragma: no cover
            yield
        raise ProcessError("Process failed")

    # receive_response should be a method that returns the generator when called
    mock_client.receive_response = lambda: process_error_response()
    return mock_client


@pytest.fixture
def mock_sdk_client_json_decode_error() -> AsyncMock:
    """Create a mock Claude SDK client that raises CLIJSONDecodeError."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    async def json_error_response() -> AsyncGenerator[Any, None]:
        # Need yield to make it a proper async generator before raising
        if False:  # pragma: no cover
            yield
        raise CLIJSONDecodeError("invalid json", Exception("JSON decode failed"))

    # receive_response should be a method that returns the generator when called
    mock_client.receive_response = lambda: json_error_response()
    return mock_client


@pytest.fixture
def mock_sdk_client_unexpected_error() -> AsyncMock:
    """Create a mock Claude SDK client that raises unexpected error."""
    mock_client = AsyncMock()
    mock_client.connect = AsyncMock()
    mock_client.disconnect = AsyncMock()
    mock_client.query = AsyncMock()

    async def unexpected_error_response() -> AsyncGenerator[Any, None]:
        # Need yield to make it a proper async generator before raising
        if False:  # pragma: no cover
            yield
        raise ValueError("Unexpected error")

    # receive_response should be a method that returns the generator when called
    mock_client.receive_response = lambda: unexpected_error_response()
    return mock_client
