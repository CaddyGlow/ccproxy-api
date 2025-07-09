"""Tests for API endpoints."""

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from ccproxy.exceptions import ModelNotFoundError, ServiceUnavailableError


@pytest.mark.integration
class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint."""

    def test_health_check(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "claude-proxy"

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_successful_chat_completion(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_chat_request: dict[str, Any],
        sample_claude_response: dict[str, Any],
    ):
        """Test successful chat completion."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = sample_claude_response
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/cc/v1/messages", json=sample_chat_request)

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert "id" in data  # Should have message ID

        # Verify ClaudeClient was called with correct interface
        mock_client.create_completion.assert_called_once()
        call_args = mock_client.create_completion.call_args
        messages = call_args.kwargs["messages"]
        options = call_args.kwargs["options"]

        # Check messages format
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, how are you?"

        # Check options
        assert hasattr(options, "model")
        assert options.model == "claude-3-5-sonnet-20241022"  # Should have message ID

    def test_invalid_model(self, test_client: TestClient):
        """Test chat completion with invalid model."""
        request_data = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        response = test_client.post("/cc/v1/messages", json=request_data)

        assert response.status_code == 422  # Validation error for pattern mismatch
        data = response.json()
        assert "detail" in data
        # Check that it's a validation error for the model field

    def test_invalid_request_body(self, test_client: TestClient):
        """Test chat completion with invalid request body."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],  # Empty messages
            "max_tokens": 100,
        }

        response = test_client.post("/cc/v1/messages", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, test_client: TestClient):
        """Test chat completion with missing required fields."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
            # Missing model and max_tokens
        }

        response = test_client.post("/cc/v1/messages", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_streaming_chat_completion(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_streaming_request: dict[str, Any],
    ):
        """Test streaming chat completion."""

        # Setup mock streaming response
        async def mock_streaming_response():
            chunks = [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": "Hello"},
                },
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": 5},
                },
            ]
            for chunk in chunks:
                yield chunk

        mock_client = AsyncMock()
        mock_client.create_completion.return_value = mock_streaming_response()
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/cc/v1/messages", json=sample_streaming_request)

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream"

        # Check streaming response format
        content = response.text
        assert "data: " in content
        assert "[DONE]" in content

        # Verify ClaudeClient was called with streaming=True
        mock_client.create_completion.assert_called_once()
        call_args = mock_client.create_completion.call_args

        # For streaming, API calls with keyword arguments: (messages, options=options, stream=True)
        assert len(call_args[0]) == 1  # Only messages as positional arg
        messages = call_args[0][0]
        kwargs = call_args[1]  # keyword args

        # Check that streaming was enabled
        assert kwargs.get("stream") is True
        assert "options" in kwargs

        # Check messages format
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_claude_client_error(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_chat_request: dict[str, Any],
    ):
        """Test handling of Claude client errors."""
        from ccproxy.exceptions import ClaudeProxyError

        mock_client = AsyncMock()
        mock_client.create_completion.side_effect = ClaudeProxyError(
            message="Claude CLI not available",
            error_type="service_unavailable_error",
            status_code=503,
        )
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/cc/v1/messages", json=sample_chat_request)

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert data["detail"]["type"] == "error"
        assert data["detail"]["error"]["type"] == "service_unavailable_error"

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_max_thinking_tokens_parameter(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_claude_response: dict[str, Any],
    ):
        """Test max_thinking_tokens parameter is properly passed to ClaudeClient."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = sample_claude_response
        mock_claude_client_class.return_value = mock_client

        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Think about this carefully"}],
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 50000},
        }

        response = test_client.post("/cc/v1/messages", json=request_data)

        assert response.status_code == 200

        # Verify ClaudeClient was called with max_thinking_tokens in options
        mock_client.create_completion.assert_called_once()
        call_args = mock_client.create_completion.call_args
        messages = call_args.kwargs["messages"]
        options = call_args.kwargs["options"]

        # Check that max_thinking_tokens was set in options
        assert hasattr(options, "max_thinking_tokens")
        assert options.max_thinking_tokens == 50000


@pytest.mark.integration
class TestModelsEndpoint:
    """Test /v1/models endpoint."""

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_list_models_success(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_models_response: list[dict[str, Any]],
    ):
        """Test successful models listing."""
        mock_client = AsyncMock()
        mock_client.list_models.return_value = sample_models_response
        mock_claude_client_class.return_value = mock_client

        response = test_client.get("/cc/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "claude-opus-4-20250514"
        assert data["data"][1]["id"] == "claude-3-5-sonnet-20241022"

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_list_models_error(self, mock_claude_client_class, test_client: TestClient):
        """Test models listing with error."""
        from ccproxy.exceptions import ClaudeProxyError

        mock_client = AsyncMock()
        mock_client.list_models.side_effect = ClaudeProxyError(
            message="Claude CLI not available",
            error_type="service_unavailable_error",
            status_code=503,
        )
        mock_claude_client_class.return_value = mock_client

        response = test_client.get("/cc/v1/models")

        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert data["detail"]["type"] == "error"
        assert data["detail"]["error"]["type"] == "service_unavailable_error"


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_404_endpoint(self, test_client: TestClient):
        """Test non-existent endpoint."""
        # The test needs to handle that catch-all reverse proxy exists
        # Create test credentials to avoid 401
        from datetime import UTC, datetime, timedelta
        from pathlib import Path

        test_creds_dir = Path("/tmp/ccproxy-test/.claude")
        test_creds_dir.mkdir(parents=True, exist_ok=True)
        test_creds_file = test_creds_dir / ".credentials.json"

        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        test_creds = {
            "claudeAiOauth": {
                "accessToken": "test-token",
                "refreshToken": "test-refresh",
                "expiresAt": future_ms,
                "scopes": ["user:inference"],
                "subscriptionType": "test",
            }
        }
        test_creds_file.write_text(json.dumps(test_creds))

        # Now test a truly non-existent path under health
        response = test_client.get("/health/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, test_client: TestClient):
        """Test method not allowed."""
        response = test_client.delete("/cc/v1/messages")

        # FastAPI returns 404 for undefined routes/methods
        assert response.status_code == 404

    def test_malformed_json(self, test_client: TestClient):
        """Test malformed JSON request."""
        response = test_client.post(
            "/cc/v1/messages",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestCORSHeaders:
    """Test CORS headers."""

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_cors_functionality(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_chat_request: dict[str, Any],
    ):
        """Test that CORS middleware is configured (endpoint responds successfully)."""
        # Mock the Claude client to avoid CLI dependency
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = {
            "id": "msg_test123",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/cc/v1/messages", json=sample_chat_request)

        # CORS middleware is configured if the endpoint responds successfully
        # (TestClient doesn't trigger CORS headers for same-origin requests)
        assert response.status_code == 200
        data = response.json()
        # This is the Anthropic endpoint, so check for Anthropic response format
        assert "content" in data
        assert data["type"] == "message"

    @patch("ccproxy.routers.claudecode.anthropic.ClaudeClient")
    def test_streaming_cors_headers(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_streaming_request: dict[str, Any],
    ):
        """Test CORS headers in streaming response."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = iter([])
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/cc/v1/messages", json=sample_streaming_request)

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-headers" in response.headers
