"""Tests for API endpoints."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_proxy.exceptions import ModelNotFoundError, ServiceUnavailableError


class TestChatCompletionsEndpoint:
    """Test /v1/chat/completions endpoint."""

    def test_health_check(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "claude-proxy"

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_successful_chat_completion(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_chat_request: dict,
        sample_claude_response: dict,
    ):
        """Test successful chat completion."""
        # Setup mock
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = sample_claude_response
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/v1/chat/completions", json=sample_chat_request)

        assert response.status_code == 200
        data = response.json()
        assert data["type"] == "message"
        assert data["role"] == "assistant"
        assert data["model"] == "claude-3-5-sonnet-20241022"
        assert "id" in data  # Should have message ID

    def test_invalid_model(self, test_client: TestClient):
        """Test chat completion with invalid model."""
        request_data = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
        }

        response = test_client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 404
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "not_found_error"
        assert "not found" in data["error"]["message"]

    def test_invalid_request_body(self, test_client: TestClient):
        """Test chat completion with invalid request body."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],  # Empty messages
            "max_tokens": 100,
        }

        response = test_client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_missing_required_fields(self, test_client: TestClient):
        """Test chat completion with missing required fields."""
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
            # Missing model and max_tokens
        }

        response = test_client.post("/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_streaming_chat_completion(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_streaming_request: dict,
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

        response = test_client.post(
            "/v1/chat/completions", json=sample_streaming_request
        )

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

        # Check streaming response format
        content = response.text
        assert "data: " in content
        assert "[DONE]" in content

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_claude_client_error(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_chat_request: dict,
    ):
        """Test handling of Claude client errors."""
        mock_client = AsyncMock()
        mock_client.create_completion.side_effect = ServiceUnavailableError(
            "Claude CLI not available"
        )
        mock_claude_client_class.return_value = mock_client

        response = test_client.post("/v1/chat/completions", json=sample_chat_request)

        assert response.status_code == 503
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "service_unavailable_error"


class TestModelsEndpoint:
    """Test /v1/models endpoint."""

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_list_models_success(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_models_response: list,
    ):
        """Test successful models listing."""
        mock_client = AsyncMock()
        mock_client.list_models.return_value = sample_models_response
        mock_claude_client_class.return_value = mock_client

        response = test_client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert "data" in data
        assert len(data["data"]) == 2
        assert data["data"][0]["id"] == "claude-3-opus-20240229"
        assert data["data"][1]["id"] == "claude-3-5-sonnet-20241022"

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_list_models_error(self, mock_claude_client_class, test_client: TestClient):
        """Test models listing with error."""
        mock_client = AsyncMock()
        mock_client.list_models.side_effect = ServiceUnavailableError(
            "Claude CLI not available"
        )
        mock_claude_client_class.return_value = mock_client

        response = test_client.get("/v1/models")

        assert response.status_code == 503
        data = response.json()
        assert data["type"] == "error"
        assert data["error"]["type"] == "service_unavailable_error"


class TestErrorHandling:
    """Test error handling across endpoints."""

    def test_404_endpoint(self, test_client: TestClient):
        """Test non-existent endpoint."""
        response = test_client.get("/nonexistent")

        assert response.status_code == 404

    def test_method_not_allowed(self, test_client: TestClient):
        """Test method not allowed."""
        response = test_client.delete("/v1/chat/completions")

        assert response.status_code == 405

    def test_malformed_json(self, test_client: TestClient):
        """Test malformed JSON request."""
        response = test_client.post(
            "/v1/chat/completions",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422


class TestCORSHeaders:
    """Test CORS headers."""

    def test_cors_headers_present(self, test_client: TestClient):
        """Test that CORS headers are present."""
        response = test_client.options("/v1/chat/completions")

        # Should have CORS headers due to middleware
        assert "access-control-allow-origin" in response.headers

    @patch("claude_proxy.api.v1.chat.ClaudeClient")
    def test_streaming_cors_headers(
        self,
        mock_claude_client_class,
        test_client: TestClient,
        sample_streaming_request: dict,
    ):
        """Test CORS headers in streaming response."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = iter([])
        mock_claude_client_class.return_value = mock_client

        response = test_client.post(
            "/v1/chat/completions", json=sample_streaming_request
        )

        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-headers" in response.headers
