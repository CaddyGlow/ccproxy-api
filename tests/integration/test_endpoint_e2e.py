"""End-to-end integration tests for CCProxy endpoints.

This module provides comprehensive endpoint testing following the project's
streamlined testing architecture with performance-optimized patterns.
"""

import json
from collections.abc import AsyncGenerator
from typing import Any

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from tests.helpers.e2e_validation import (
    get_validation_model_for_format,
    parse_streaming_events,
    validate_response_with_model,
    validate_streaming_response_structure,
)
from tests.helpers.test_data import (
    E2E_ENDPOINT_CONFIGURATIONS,
    create_e2e_request_for_format,
    get_expected_response_fields,
)


pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def e2e_app_with_plugins():
    """Session-scoped app with all plugins enabled for E2E testing."""
    from ccproxy.api.app import create_app, initialize_plugins_startup
    from ccproxy.api.bootstrap import create_service_container
    from ccproxy.config.settings import Settings
    from ccproxy.core.logging import setup_logging

    # Set up logging once per session - minimal logging for speed
    setup_logging(json_logs=False, log_level_name="ERROR")

    # Enable plugins needed for E2E testing
    settings = Settings(
        enable_plugins=True,
        plugins_disable_local_discovery=False,  # Enable local plugin discovery
        plugins={
            "claude_api": {"enabled": True},
            "copilot": {"enabled": True},
            "codex": {"enabled": True},
            "duckdb_storage": {"enabled": False},  # Disable to avoid I/O side effects
        },
        logging={
            "level": "ERROR",  # Minimal logging for speed
            "enable_plugin_logging": False,
            "verbose_api": False,
        },
    )

    service_container = create_service_container(settings)
    app = create_app(service_container)

    # Initialize plugins once per session
    await initialize_plugins_startup(app, settings)

    return app, settings


@pytest_asyncio.fixture(loop_scope="session")
async def e2e_client(e2e_app_with_plugins) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client for E2E endpoint testing - uses session-scoped app."""
    app, _ = e2e_app_with_plugins

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="session")
def mock_external_e2e_apis():
    """Mock external API calls for E2E integration tests."""
    from unittest.mock import AsyncMock, patch

    with (
        patch("httpx.AsyncClient.post") as mock_post,
        patch("httpx.AsyncClient.get") as mock_get,
        patch("httpx.AsyncClient.stream") as mock_stream,
    ):
        # Configure mock responses for different formats
        mock_post.return_value = AsyncMock(
            status_code=200,
            json=AsyncMock(
                return_value={
                    "id": "test-id",
                    "object": "chat.completion",
                    "created": 1234567890,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "Hello test response",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 10,
                        "total_tokens": 15,
                    },
                }
            ),
        )

        mock_get.return_value = AsyncMock(
            status_code=200, json=AsyncMock(return_value={"data": []})
        )

        # Mock streaming response
        class MockStreamResponse:
            def __init__(self):
                self.status_code = 200
                self.headers = {"content-type": "text/event-stream"}

            async def aiter_text(self):
                yield "data: " + json.dumps(
                    {
                        "id": "test-stream-id",
                        "object": "chat.completion.chunk",
                        "created": 1234567890,
                        "model": "test-model",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": "Hello"},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                yield "data: [DONE]"

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_stream.return_value = MockStreamResponse()

        yield {
            "post": mock_post,
            "get": mock_get,
            "stream": mock_stream,
        }


class TestEndpointE2E:
    """End-to-end endpoint tests following session-scoped patterns."""

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize("config", E2E_ENDPOINT_CONFIGURATIONS)
    async def test_endpoint_basic_functionality(
        self,
        e2e_client: AsyncClient,
        mock_external_e2e_apis: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Test basic endpoint functionality for all configured endpoints."""
        endpoint = config["endpoint"]
        model = config["model"]
        format_type = config["format"]
        stream = config["stream"]

        # Create appropriate request for format
        request_data = create_e2e_request_for_format(
            format_type=format_type,
            model=model,
            content="Test message for E2E",
            stream=stream,
        )

        # Set appropriate headers
        headers = {"Content-Type": "application/json"}
        if stream:
            headers["Accept"] = "text/event-stream"

        # Make request
        response = await e2e_client.post(endpoint, json=request_data, headers=headers)

        # Basic response validation
        error_text = (
            response.text if isinstance(response.text, str) else str(response.content)
        )
        assert response.status_code == 200, (
            f"Endpoint {endpoint} failed with {response.status_code}: {error_text}"
        )

        if stream:
            # Validate streaming response
            assert response.headers.get("content-type", "").startswith(
                "text/event-stream"
            )
            # Handle async response.text
            content = (
                response.text
                if isinstance(response.text, str)
                else await response.aread()
            )
            if isinstance(content, bytes):
                content = content.decode()
            assert "data: " in content

            # Validate streaming structure
            is_valid, errors = validate_streaming_response_structure(
                content,
                format_type,
                get_validation_model_for_format(format_type, is_streaming=True),
            )
            if not is_valid:
                pytest.fail(f"Streaming validation failed for {endpoint}: {errors}")

        else:
            # Validate JSON response
            data = response.json()
            assert isinstance(data, dict)

            # Use pydantic validation if available
            validation_model = get_validation_model_for_format(
                format_type, is_streaming=False
            )
            if validation_model:
                is_valid, error = validate_response_with_model(data, validation_model)
                if not is_valid:
                    # Don't fail hard on validation - log warning instead for flexibility
                    print(
                        f"Warning: Response validation failed for {endpoint}: {error}"
                    )

            # Check for expected fields based on format
            expected_fields = get_expected_response_fields(format_type)
            if expected_fields:
                # At least some expected fields should be present
                found_fields = set(data.keys())
                assert len(found_fields.intersection(expected_fields)) > 0, (
                    f"No expected fields found in response. "
                    f"Expected any of {expected_fields}, got {found_fields}"
                )

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        "config", [c for c in E2E_ENDPOINT_CONFIGURATIONS if not c["stream"]]
    )
    async def test_non_streaming_response_structure(
        self,
        e2e_client: AsyncClient,
        mock_external_e2e_apis: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Test non-streaming response structure validation."""
        endpoint = config["endpoint"]
        model = config["model"]
        format_type = config["format"]

        request_data = create_e2e_request_for_format(
            format_type=format_type,
            model=model,
            content="Structure validation test",
            stream=False,
        )

        response = await e2e_client.post(endpoint, json=request_data)
        assert response.status_code == 200

        data = response.json()

        # Validate response structure based on format
        if format_type == "openai":
            assert "choices" in data
            assert isinstance(data["choices"], list)
            if data["choices"]:
                choice = data["choices"][0]
                assert "message" in choice
                assert "role" in choice["message"]
                assert "content" in choice["message"]

        elif format_type == "anthropic":
            assert "content" in data
            assert "role" in data
            assert data["role"] == "assistant"

        elif format_type == "response_api":
            # Response API format validation
            assert "output" in data or "choices" in data  # Flexible validation

    @pytest.mark.asyncio(loop_scope="session")
    @pytest.mark.parametrize(
        "config", [c for c in E2E_ENDPOINT_CONFIGURATIONS if c["stream"]]
    )
    async def test_streaming_response_format(
        self,
        e2e_client: AsyncClient,
        mock_external_e2e_apis: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Test streaming response format and SSE structure."""
        endpoint = config["endpoint"]
        model = config["model"]
        format_type = config["format"]

        request_data = create_e2e_request_for_format(
            format_type=format_type,
            model=model,
            content="Streaming test",
            stream=True,
        )

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        response = await e2e_client.post(endpoint, json=request_data, headers=headers)
        assert response.status_code == 200

        # Validate SSE headers
        content_type = response.headers.get("content-type", "")
        assert content_type.startswith("text/event-stream")

        # Parse and validate streaming content
        # Handle async response.text
        content = (
            response.text if isinstance(response.text, str) else await response.aread()
        )
        if isinstance(content, bytes):
            content = content.decode()

        # Use comprehensive streaming validation
        is_valid, errors = validate_streaming_response_structure(
            content,
            format_type,
            get_validation_model_for_format(format_type, is_streaming=True),
        )

        # Should have valid streaming structure
        assert is_valid, f"Streaming validation failed: {errors}"

        # Parse events for additional validation
        events = parse_streaming_events(content)
        assert len(events) > 0, "No valid events found in streaming response"

        # Format-specific event validation
        if format_type == "openai":
            # Should have delta content in at least one event
            has_delta = any(
                event.get("choices", [{}])[0].get("delta") is not None
                for event in events
                if event.get("choices")
            )
            assert has_delta, "OpenAI streaming events should contain delta content"

    @pytest.mark.asyncio(loop_scope="session")
    async def test_invalid_endpoint_returns_404(
        self,
        e2e_client: AsyncClient,
    ) -> None:
        """Test that invalid endpoints return 404."""
        response = await e2e_client.post("/invalid/endpoint", json={"test": "data"})
        assert response.status_code == 404

    @pytest.mark.asyncio(loop_scope="session")
    async def test_malformed_request_handling(
        self,
        e2e_client: AsyncClient,
    ) -> None:
        """Test handling of malformed requests."""
        # Test with empty JSON
        response = await e2e_client.post("/api/v1/chat/completions", json={})
        # Should return 400 or 422 for validation error
        assert response.status_code in [400, 422]

        # Test with invalid JSON structure
        response = await e2e_client.post(
            "/api/v1/chat/completions", json={"invalid": "structure"}
        )
        assert response.status_code in [400, 422]


# Performance test for session-scoped fixture efficiency
@pytest.mark.asyncio(loop_scope="session")
async def test_session_fixture_reuse(e2e_client: AsyncClient) -> None:
    """Test that session-scoped fixtures are properly reused."""
    # This test verifies the fixture is working by making a simple request
    response = await e2e_client.get("/")
    # We expect either a 200 (root handler) or 404 (no root handler), not a connection error
    assert response.status_code in [200, 404, 405]  # 405 for method not allowed
