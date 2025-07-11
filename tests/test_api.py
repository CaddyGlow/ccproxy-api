"""API endpoint tests for both OpenAI and Anthropic formats.

Tests all HTTP endpoints, request/response validation, authentication,
and error handling without mocking internal components.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from pytest_httpx import HTTPXMock


class TestOpenAIEndpoints:
    """Test OpenAI-compatible API endpoints."""

    def test_chat_completions_success(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test successful OpenAI chat completion request."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI response format
        assert "id" in data
        assert "object" in data
        assert "created" in data
        assert "model" in data
        assert "choices" in data
        assert "usage" in data

        # Verify choice structure
        choice = data["choices"][0]
        assert "index" in choice
        assert "message" in choice
        assert "finish_reason" in choice

        # Verify message structure
        message = choice["message"]
        assert message["role"] == "assistant"
        assert "content" in message

    def test_chat_completions_with_system_message(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with system message."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            "max_tokens": 50,
        }

        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_invalid_model(
        self,
        client: TestClient,
    ) -> None:
        """Test OpenAI chat completion with invalid model."""

        request_data = {
            "model": "invalid-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
        }

        response = client.post("/sdk/v1/chat/completions", json=request_data)

        assert response.status_code == 400
        data = response.json()
        assert "error" in data

    def test_chat_completions_missing_messages(self, client: TestClient) -> None:
        """Test OpenAI chat completion with missing messages."""
        request_data = {"model": "claude-3-5-sonnet-20241022", "max_tokens": 50}

        response = client.post("/sdk/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_chat_completions_empty_messages(self, client: TestClient) -> None:
        """Test OpenAI chat completion with empty messages array."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [],
            "max_tokens": 50,
        }

        response = client.post("/sdk/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_chat_completions_malformed_message(self, client: TestClient) -> None:
        """Test OpenAI chat completion with malformed message."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"invalid_field": "user", "content": "Hello"}],
            "max_tokens": 50,
        }

        response = client.post("/sdk/v1/chat/completions", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_list_models_openai(
        self, client: TestClient, claude_responses: dict[str, Any]
    ) -> None:
        """Test OpenAI models list endpoint."""
        response = client.get("/openai/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Verify OpenAI models response format
        assert "object" in data
        assert data["object"] == "list"
        assert "data" in data
        assert isinstance(data["data"], list)

        # Verify model entries
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model

    def test_openai_status(self, client: TestClient) -> None:
        """Test OpenAI status endpoint."""
        response = client.get("/openai/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestAnthropicEndpoints:
    """Test Anthropic-compatible API endpoints."""

    def test_create_message_success(self, client_with_mock_claude: TestClient) -> None:
        """Test successful Anthropic message creation."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello, Claude!"}],
        }

        response = client_with_mock_claude.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify Anthropic response format
        assert "id" in data
        assert "type" in data
        assert data["type"] == "message"
        assert "role" in data
        assert data["role"] == "assistant"
        assert "content" in data
        assert "model" in data
        assert "stop_reason" in data
        assert "usage" in data

    def test_create_message_with_system(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic message creation with system message."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "Hello!"}],
        }

        response = client_with_mock_claude.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["role"] == "assistant"

    def test_create_message_invalid_model(
        self,
        client: TestClient,
    ) -> None:
        """Test Anthropic message creation with invalid model."""

        request_data = {
            "model": "invalid-model",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data

    def test_create_message_missing_max_tokens(self, client: TestClient) -> None:
        """Test Anthropic message creation with missing max_tokens."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_create_message_invalid_message_role(self, client: TestClient) -> None:
        """Test Anthropic message creation with invalid role."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [{"role": "invalid", "content": "Hello"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 422  # Validation error

    def test_list_models_anthropic(self, client: TestClient) -> None:
        """Test Anthropic models list endpoint."""
        response = client.get("/v1/models")

        assert response.status_code == 200
        data = response.json()

        # Verify Anthropic models response format
        assert "data" in data
        assert isinstance(data["data"], list)

        # Verify model entries
        if data["data"]:
            model = data["data"][0]
            assert "id" in model
            assert "object" in model
            assert "created" in model
            assert "owned_by" in model

    def test_anthropic_status(self, client: TestClient) -> None:
        """Test Anthropic status endpoint."""
        response = client.get("/v1/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestClaudeSDKEndpoints:
    """Test Claude SDK direct endpoints."""

    def test_claude_status(self, client: TestClient) -> None:
        """Test Claude SDK status endpoint."""
        response = client.get("/sdk/status")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestDualOpenAIEndpoints:
    """Test that both SDK and API OpenAI endpoints work correctly."""

    def test_chat_completions_both_paths(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test that OpenAI endpoints work with Claude SDK."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello, world!"}],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        # Test /sdk/v1 path (Claude SDK) - primary working path
        response_sdk = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=request_data
        )
        assert response_sdk.status_code == 200
        data_sdk = response_sdk.json()

        # Response should have OpenAI format
        assert "id" in data_sdk
        assert "object" in data_sdk
        assert "created" in data_sdk
        assert "model" in data_sdk
        assert "choices" in data_sdk
        assert "usage" in data_sdk

    def test_models_list_both_paths(self, client: TestClient) -> None:
        """Test that models endpoint works on both OpenAI paths."""
        # Test /openai/v1/models
        response_openai = client.get("/openai/v1/models")
        assert response_openai.status_code == 200
        data_openai = response_openai.json()

        # Test /v1/models (should return Anthropic format, but still work)
        response_v1 = client.get("/v1/models")
        assert response_v1.status_code == 200
        data_v1 = response_v1.json()

        # OpenAI endpoint should have OpenAI format
        assert "object" in data_openai
        assert data_openai["object"] == "list"
        assert "data" in data_openai

        # V1 endpoint returns Anthropic format
        assert "data" in data_v1


class TestAuthenticationEndpoints:
    """Test API endpoints with authentication."""

    def test_openai_chat_completions_authenticated(
        self,
        client_with_auth: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test authenticated OpenAI chat completion."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
        }

        response = client_with_auth.post(
            "/openai/v1/chat/completions", json=request_data, headers=auth_headers
        )

        assert response.status_code == 200

    def test_openai_chat_completions_unauthenticated(
        self, client_with_auth: TestClient
    ) -> None:
        """Test OpenAI chat completion endpoint returns deprecation message."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
        }

        response = client_with_auth.post(
            "/openai/v1/chat/completions", json=request_data
        )

        # Deprecated endpoint returns 200 with deprecation message
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "deprecated" in data["error"]["message"].lower()

    def test_openai_chat_completions_invalid_token(
        self, client_with_auth: TestClient
    ) -> None:
        """Test OpenAI chat completion endpoint returns deprecation message regardless of token."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 50,
        }

        response = client_with_auth.post(
            "/openai/v1/chat/completions",
            json=request_data,
            headers={"Authorization": "Bearer invalid-token"},
        )

        # Deprecated endpoint returns 200 with deprecation message regardless of auth
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "deprecated" in data["error"]["message"].lower()

    def test_anthropic_messages_authenticated(
        self,
        client_with_auth: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test authenticated Anthropic message creation."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client_with_auth.post(
            "/v1/messages", json=request_data, headers=auth_headers
        )

        assert response.status_code == 200

    def test_anthropic_messages_unauthenticated(
        self, client_with_auth: TestClient
    ) -> None:
        """Test Anthropic messages endpoint returns deprecation message."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client_with_auth.post("/v1/messages", json=request_data)

        # Deprecated endpoint returns 200 with deprecation message
        assert response.status_code == 200
        data = response.json()
        assert "error" in data
        assert "deprecated" in data["error"]["message"].lower()

    def test_models_list_authenticated(
        self, client_with_auth: TestClient, auth_headers: dict[str, str]
    ) -> None:
        """Test models list endpoints with authentication."""
        # Test OpenAI models endpoint
        response = client_with_auth.get("/openai/v1/models", headers=auth_headers)
        assert response.status_code == 200

        # Test Anthropic models endpoint
        response = client_with_auth.get("/v1/models", headers=auth_headers)
        assert response.status_code == 200

    def test_models_list_unauthenticated(self, client_with_auth: TestClient) -> None:
        """Test models list endpoints return model data."""
        # Test OpenAI models endpoint - should return model list
        response = client_with_auth.get("/openai/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data

        # Test Anthropic models endpoint - should return model list
        response = client_with_auth.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert "data" in data


class TestErrorHandling:
    """Test API error handling and edge cases."""

    def test_rate_limit_error(
        self,
        client: TestClient,
    ) -> None:
        """Test handling when Claude CLI is not available."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        # Without Claude CLI, should return 503 Service Unavailable
        assert response.status_code == 503
        data = response.json()
        assert "error" in data

    def test_server_error(self, client: TestClient) -> None:
        """Test handling when Claude CLI is not available."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        # Without Claude CLI, should return 503 Service Unavailable
        assert response.status_code == 503

    def test_invalid_json(self, client: TestClient) -> None:
        """Test handling of invalid JSON requests."""
        response = client.post(
            "/sdk/v1/messages",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 422

    def test_unsupported_content_type(self, client: TestClient) -> None:
        """Test handling of unsupported content types."""
        response = client.post(
            "/sdk/v1/messages",
            content="some data",
            headers={"Content-Type": "text/plain"},
        )

        assert response.status_code == 422

    def test_large_request_body(self, client: TestClient) -> None:
        """Test handling of large request bodies."""
        # Create a very large message
        large_content = "x" * 1000000  # 1MB of text

        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": large_content}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        # Without Claude CLI, should return 503 Service Unavailable
        assert response.status_code == 503

    def test_malformed_headers(self, client: TestClient) -> None:
        """Test handling of malformed headers."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Hello"}],
        }

        # Test with invalid authorization header format
        response = client.post(
            "/sdk/v1/messages",
            json=request_data,
            headers={"Authorization": "InvalidFormat"},
        )

        # Without Claude CLI, should return 503 Service Unavailable
        assert response.status_code == 503


class TestResponseValidation:
    """Test API response validation and format consistency."""

    def test_openai_response_schema(self, client_with_mock_claude: TestClient) -> None:
        """Test OpenAI response follows correct schema."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 50,
        }

        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=request_data
        )

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = ["id", "object", "created", "model", "choices", "usage"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify types
        assert isinstance(data["id"], str)
        assert isinstance(data["object"], str)
        assert isinstance(data["created"], int)
        assert isinstance(data["model"], str)
        assert isinstance(data["choices"], list)
        assert isinstance(data["usage"], dict)

    def test_anthropic_response_schema(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic response follows correct schema."""
        request_data = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client_with_mock_claude.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 200
        data = response.json()

        # Verify required fields
        required_fields = [
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "usage",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"

        # Verify types
        assert isinstance(data["id"], str)
        assert isinstance(data["type"], str)
        assert isinstance(data["role"], str)
        assert isinstance(data["content"], list)
        assert isinstance(data["model"], str)
        assert isinstance(data["usage"], dict)

    def test_error_response_schema(
        self,
        client: TestClient,
        claude_responses: dict[str, Any],
        httpx_mock: HTTPXMock,
    ) -> None:
        """Test error responses follow correct schema."""
        # Mock error response
        httpx_mock.add_response(
            url="https://api.anthropic.com/v1/messages",
            json=claude_responses["error_response"],
            status_code=400,
        )

        request_data = {
            "model": "invalid-model",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Test"}],
        }

        response = client.post("/sdk/v1/messages", json=request_data)

        assert response.status_code == 400
        data = response.json()

        # Verify error structure
        assert "error" in data
        error = data["error"]
        assert "type" in error
        assert "message" in error
        assert isinstance(error["type"], str)
        assert isinstance(error["message"], str)


class TestStatusEndpoints:
    """Test various status and health check endpoints."""

    def test_all_status_endpoints(self, client: TestClient) -> None:
        """Test all status endpoints return successfully."""
        status_endpoints = [
            "/v1/status",
            "/openai/v1/status",
            "/sdk/status",
            "/health",
            "/health/ready",
            "/health/live",
        ]

        for endpoint in status_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Status endpoint {endpoint} failed"

            data = response.json()
            assert "status" in data or "message" in data or "health" in data


@pytest.mark.unit
class TestRequestValidation:
    """Test request validation without external calls."""

    def test_openai_request_validation(self, client: TestClient) -> None:
        """Test OpenAI request validation rules."""
        # Test missing model
        response = client.post(
            "/openai/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 422

        # Test missing messages
        response = client.post(
            "/openai/v1/chat/completions", json={"model": "claude-3-5-sonnet-20241022"}
        )
        assert response.status_code == 422

        # Test invalid message role
        response = client.post(
            "/openai/v1/chat/completions",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "invalid", "content": "test"}],
            },
        )
        assert response.status_code == 422

    def test_anthropic_request_validation(self, client: TestClient) -> None:
        """Test Anthropic request validation rules."""
        # Test missing model
        response = client.post(
            "/v1/messages",
            json={"max_tokens": 50, "messages": [{"role": "user", "content": "test"}]},
        )
        assert response.status_code == 422

        # Test missing max_tokens
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 422

        # Test invalid max_tokens
        response = client.post(
            "/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": -1,
                "messages": [{"role": "user", "content": "test"}],
            },
        )
        assert response.status_code == 422
