"""Shared test fixtures and configuration for ccproxy tests.

This module provides minimal, focused fixtures for testing the ccproxy application.
All fixtures have proper type hints and are designed to work with real components
while mocking only external services.
"""

import json
from collections.abc import AsyncGenerator, Generator

# Override settings for testing
from functools import lru_cache
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pytest_httpx import HTTPXMock

from ccproxy.api.app import create_app
from ccproxy.config.security import SecuritySettings
from ccproxy.config.server import ServerSettings
from ccproxy.config.settings import Settings, get_settings
from ccproxy.metrics.storage.memory import InMemoryMetricsStorage


@lru_cache
def get_test_settings() -> Settings:
    """Get test settings - overrides the default settings provider."""
    return Settings(
        server=ServerSettings(log_level="WARNING"),
        security=SecuritySettings(auth_token=None),  # Disable auth by default
    )


# Test data directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Create isolated test settings with temp directories.

    Returns a Settings instance configured for testing with:
    - Temporary config and cache directories
    - In-memory metrics storage
    - No authentication by default
    - Test environment enabled
    """
    return Settings(
        server=ServerSettings(log_level="WARNING"),
        security=SecuritySettings(auth_token=None),  # No auth by default
    )


@pytest.fixture
def app(test_settings: Settings) -> FastAPI:
    """Create test FastAPI application with test settings.

    Returns a configured FastAPI app ready for testing.
    """
    # Create app
    app = create_app(settings=test_settings)

    # Override the settings dependency for testing
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: test_settings

    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create synchronous test client for API testing.

    Returns a TestClient for making synchronous HTTP requests.
    """
    return TestClient(app)


@pytest_asyncio.fixture
async def async_client(app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """Create asynchronous test client for async API testing.

    Yields an AsyncClient for making asynchronous HTTP requests.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture
def auth_settings(tmp_path: Path) -> Settings:
    """Create settings with authentication enabled.

    Returns Settings configured with a test auth token.
    """
    return Settings(
        security=SecuritySettings(auth_token="test-token-12345"),
    )


@pytest.fixture
def app_with_auth(auth_settings: Settings) -> FastAPI:
    """Create app with authentication enabled.

    Returns a FastAPI app that requires authentication.
    """
    # Create app
    app = create_app(settings=auth_settings)

    # Override the settings dependency for testing
    from ccproxy.config.settings import get_settings as original_get_settings

    app.dependency_overrides[original_get_settings] = lambda: auth_settings

    return app


@pytest.fixture
def client_with_auth(app_with_auth: FastAPI) -> TestClient:
    """Create client for testing authenticated endpoints.

    Returns a TestClient with auth-enabled app.
    """
    return TestClient(app_with_auth)


@pytest.fixture
def claude_responses() -> dict[str, Any]:
    """Load standard Claude API responses from fixtures.

    Returns a dictionary of mock Claude API responses.
    """
    responses_file = FIXTURES_DIR / "responses.json"
    if responses_file.exists():
        response_data = json.loads(responses_file.read_text())
        return response_data  # type: ignore[no-any-return]

    # Default responses if file doesn't exist yet
    return {
        "standard_completion": {
            "id": "msg_01234567890",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello! How can I help you?"}],
            "model": "claude-3-5-sonnet-20241022",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 8},
        },
        "error_response": {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified",
            },
        },
    }


@pytest.fixture
def mock_claude(httpx_mock: HTTPXMock, claude_responses: dict[str, Any]) -> HTTPXMock:
    """Mock Claude API responses for standard completion.

    Returns HTTPXMock configured with Claude API responses.
    """
    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        json=claude_responses["standard_completion"],
        status_code=200,
        headers={"content-type": "application/json"},
    )
    return httpx_mock


@pytest.fixture
def mock_claude_stream(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock Claude API streaming responses.

    Returns HTTPXMock configured for SSE streaming.
    """

    def stream_generator() -> Generator[str, None, None]:
        """Generate SSE formatted streaming response."""
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
            yield f"data: {json.dumps(event)}\n\n"

    httpx_mock.add_response(
        url="https://api.anthropic.com/v1/messages",
        content=b"".join(chunk.encode() for chunk in stream_generator()),
        status_code=200,
        headers={
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
        },
    )
    return httpx_mock


@pytest.fixture
def mock_oauth(httpx_mock: HTTPXMock) -> HTTPXMock:
    """Mock OAuth token endpoints.

    Returns HTTPXMock configured with OAuth responses.
    """
    # Mock token exchange
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/token",
        json={
            "access_token": "test_access_token",
            "refresh_token": "test_refresh_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    # Mock token refresh
    httpx_mock.add_response(
        url="https://api.anthropic.com/oauth/refresh",
        json={
            "access_token": "new_test_access_token",
            "expires_in": 3600,
            "token_type": "Bearer",
        },
        status_code=200,
    )

    return httpx_mock


@pytest.fixture
def metrics_storage() -> InMemoryMetricsStorage:
    """Create isolated in-memory metrics storage.

    Returns an InMemoryMetricsStorage instance for testing.
    """
    return InMemoryMetricsStorage()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Standard authentication headers for testing.

    Returns headers with test auth token.
    """
    return {"Authorization": "Bearer test-token-12345"}


# Pytest configuration
def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom settings."""
    # Ensure async tests work properly
    config.option.asyncio_mode = "auto"


# Test directory validation
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers."""
    for item in items:
        # Auto-mark async tests
        if "async" in item.nodeid:
            item.add_marker(pytest.mark.asyncio)

        # Add unit marker to tests not marked as real_api
        if not any(marker.name == "real_api" for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
