"""Pytest configuration and fixtures for Claude Proxy tests."""

import asyncio
import os
from collections.abc import AsyncGenerator, Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ccproxy.config.settings import Settings
from ccproxy.main import create_app
from ccproxy.services.claude_client import ClaudeClient


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test settings."""
    # Set test environment variables
    os.environ["ANTHROPIC_API_KEY"] = "test-api-key"
    # Don't set CLAUDE_CLI_PATH to avoid validation errors

    # Override global settings to disable authentication for tests
    import json

    os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps({"auth_token": None})

    # Create settings without loading any config files and disable authentication
    return Settings(_env_file=None, auth_token=None)


@pytest.fixture
def test_client(test_settings: Settings) -> TestClient:
    """Create a test client."""
    app = create_app(test_settings)
    return TestClient(app)


@pytest.fixture
def mock_claude_client() -> MagicMock:
    """Create a mock Claude client."""
    mock_client = MagicMock(spec=ClaudeClient)
    mock_client.create_completion = AsyncMock()
    mock_client.list_models = AsyncMock()
    return mock_client


@pytest.fixture
def sample_chat_request() -> dict[str, Any]:
    """Sample chat completion request."""
    return {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def sample_streaming_request() -> dict[str, Any]:
    """Sample streaming chat completion request."""
    return {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "max_tokens": 200,
        "temperature": 0.8,
        "stream": True,
    }


@pytest.fixture
def sample_claude_response() -> dict[str, Any]:
    """Sample Claude response."""
    return {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hello! I'm doing well, thank you for asking."}
        ],
        "model": "claude-3-5-sonnet-20241022",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 15, "total_tokens": 25},
    }


@pytest.fixture
def sample_streaming_response():
    """Sample streaming Claude response chunks."""

    async def _create_stream() -> AsyncGenerator[dict[str, Any], None]:
        chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test123",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-3-5-sonnet-20241022",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Once"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " upon"},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": " a time..."},
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 15},
            },
        ]

        for chunk in chunks:
            yield chunk

    return _create_stream()


@pytest.fixture
def sample_models_response() -> list[dict[str, Any]]:
    """Sample models list response."""
    return [
        {
            "id": "claude-opus-4-20250514",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
        {
            "id": "claude-3-5-sonnet-20241022",
            "object": "model",
            "created": 1677610602,
            "owned_by": "anthropic",
        },
    ]


# Cleanup after tests
@pytest.fixture(autouse=True)
def cleanup_env():
    """Clean up environment variables after each test."""
    yield
    # Clean up test environment variables
    test_vars = ["ANTHROPIC_API_KEY", "CLAUDE_CLI_PATH", "CCPROXY_CONFIG_OVERRIDES"]
    for var in test_vars:
        if var in os.environ:
            del os.environ[var]


@pytest.fixture
def mock_claude_streaming_response():
    """Mock Claude streaming response for testing."""

    async def stream_generator():
        # Message start
        yield {
            "type": "message_start",
            "message": {"id": "msg_123", "role": "assistant"},
        }

        # Content block start
        yield {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""},
        }

        # Content deltas
        yield {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "},
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "world!"},
        }

        # Content block stop
        yield {"type": "content_block_stop"}

        # Message delta
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "usage": {"output_tokens": 10}},
        }

        # Message stop
        yield {"type": "message_stop"}

    return stream_generator()


@pytest.fixture
def mock_claude_tool_streaming_response():
    """Mock Claude streaming response with tool calls for testing."""

    async def stream_generator():
        # Message start
        yield {
            "type": "message_start",
            "message": {"id": "msg_123", "role": "assistant"},
        }

        # Tool use content block start
        yield {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "tool_123",
                "name": "get_weather",
            },
        }

        # Tool use input delta
        yield {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"location": "'},
        }
        yield {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": 'New York"}'},
        }

        # Content block stop
        yield {"type": "content_block_stop"}

        # Message delta
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "usage": {"output_tokens": 5}},
        }

        # Message stop
        yield {"type": "message_stop"}

    return stream_generator()


@pytest.fixture
def mock_claude_error_streaming_response():
    """Mock Claude streaming response that raises an error."""

    async def stream_generator():
        # Message start
        yield {
            "type": "message_start",
            "message": {"id": "msg_123", "role": "assistant"},
        }

        # Content block start
        yield {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""},
        }

        # Content delta
        yield {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "},
        }

        # Raise error
        raise Exception("Test streaming error")

    return stream_generator()


@pytest.fixture
def mock_claude_cancelled_streaming_response():
    """Mock Claude streaming response that gets cancelled."""

    async def stream_generator():
        # Message start
        yield {
            "type": "message_start",
            "message": {"id": "msg_123", "role": "assistant"},
        }

        # Content block start
        yield {
            "type": "content_block_start",
            "content_block": {"type": "text", "text": ""},
        }

        # Content delta
        yield {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello "},
        }

        # Raise cancellation
        raise asyncio.CancelledError("Test cancellation")

    return stream_generator()


@pytest.fixture
def mock_claude_empty_streaming_response():
    """Mock Claude streaming response with no content."""

    async def stream_generator():
        # Message start
        yield {
            "type": "message_start",
            "message": {"id": "msg_123", "role": "assistant"},
        }

        # Message delta (no content)
        yield {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "usage": {"output_tokens": 0}},
        }

        # Message stop
        yield {"type": "message_stop"}

    return stream_generator()
