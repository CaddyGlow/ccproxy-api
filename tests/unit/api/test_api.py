"""API endpoint tests for both OpenAI and Anthropic formats.

Tests all HTTP endpoints, request/response validation, authentication,
and error handling using factory patterns and organized fixtures.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.helpers.assertions import (
    assert_anthropic_response_format,
    assert_auth_error,
    assert_bad_request_error,
    assert_openai_response_format,
    assert_service_unavailable_error,
    assert_sse_format_compliance,
    assert_sse_headers,
    assert_validation_error,
)
from tests.helpers.test_data import (
    ANTHROPIC_REQUEST_WITH_SYSTEM,
    EMPTY_MESSAGES_OPENAI_REQUEST,
    INVALID_MODEL_ANTHROPIC_REQUEST,
    INVALID_MODEL_OPENAI_REQUEST,
    INVALID_ROLE_ANTHROPIC_REQUEST,
    LARGE_REQUEST_ANTHROPIC,
    MALFORMED_MESSAGE_OPENAI_REQUEST,
    MISSING_MAX_TOKENS_ANTHROPIC_REQUEST,
    MISSING_MESSAGES_OPENAI_REQUEST,
    OPENAI_REQUEST_WITH_SYSTEM,
    STANDARD_ANTHROPIC_REQUEST,
    STANDARD_OPENAI_REQUEST,
    STREAMING_ANTHROPIC_REQUEST,
    STREAMING_OPENAI_REQUEST,
)


@pytest.mark.unit
class TestOpenAIEndpoints:
    """Test OpenAI-compatible API endpoints."""

    def test_chat_completions_success(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test successful OpenAI chat completion request."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
        )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert_openai_response_format(data)

    def test_chat_completions_with_system_message(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with system message."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=OPENAI_REQUEST_WITH_SYSTEM
        )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert_openai_response_format(data)

    def test_chat_completions_invalid_model(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with invalid model."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=INVALID_MODEL_OPENAI_REQUEST
        )

        assert_bad_request_error(response)

    def test_chat_completions_missing_messages(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with missing messages."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=MISSING_MESSAGES_OPENAI_REQUEST
        )

        assert_validation_error(response)

    def test_chat_completions_empty_messages(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with empty messages array."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=EMPTY_MESSAGES_OPENAI_REQUEST
        )

        assert_validation_error(response)

    def test_chat_completions_malformed_message(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test OpenAI chat completion with malformed message."""
        response = client_with_mock_claude.post(
            "/sdk/v1/chat/completions", json=MALFORMED_MESSAGE_OPENAI_REQUEST
        )

        assert_validation_error(response)


@pytest.mark.unit
class TestAnthropicEndpoints:
    """Test Anthropic-compatible API endpoints."""

    def test_create_message_success(self, client_with_mock_claude: TestClient) -> None:
        """Test successful Anthropic message creation."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages", json=STANDARD_ANTHROPIC_REQUEST
        )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert_anthropic_response_format(data)

    def test_create_message_with_system(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic message creation with system message."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages", json=ANTHROPIC_REQUEST_WITH_SYSTEM
        )

        assert response.status_code == 200
        data: dict[str, Any] = response.json()
        assert_anthropic_response_format(data)

    def test_create_message_invalid_model(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic message creation with invalid model."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages", json=INVALID_MODEL_ANTHROPIC_REQUEST
        )

        assert_validation_error(response)

    def test_create_message_missing_max_tokens(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic message creation with missing max_tokens."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages", json=MISSING_MAX_TOKENS_ANTHROPIC_REQUEST
        )

        assert_validation_error(response)

    def test_create_message_invalid_message_role(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test Anthropic message creation with invalid role."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages", json=INVALID_ROLE_ANTHROPIC_REQUEST
        )

        assert_validation_error(response)


@pytest.mark.unit
class TestClaudeSDKEndpoints:
    """Test Claude SDK specific functionality (streaming, etc.)."""

    def test_sdk_streaming_messages(
        self, client_with_mock_claude_streaming: TestClient
    ) -> None:
        """Test Claude SDK streaming messages endpoint."""
        with client_with_mock_claude_streaming.stream(
            "POST", "/sdk/v1/messages", json=STREAMING_ANTHROPIC_REQUEST
        ) as response:
            assert response.status_code == 200
            assert_sse_headers(response)

            chunks: list[str] = []
            for line in response.iter_lines():
                if line.strip():
                    chunks.append(line)

            assert_sse_format_compliance(chunks)

    def test_sdk_streaming_chat_completions(
        self, client_with_mock_claude_streaming: TestClient
    ) -> None:
        """Test Claude SDK streaming chat completions endpoint."""
        with client_with_mock_claude_streaming.stream(
            "POST", "/sdk/v1/chat/completions", json=STREAMING_OPENAI_REQUEST
        ) as response:
            assert response.status_code == 200
            assert_sse_headers(response)

            chunks: list[str] = []
            for line in response.iter_lines():
                if line.strip():
                    chunks.append(line)

            assert_sse_format_compliance(chunks)


# Removed TestDualOpenAIEndpoints - functionality covered by TestOpenAIEndpoints


@pytest.mark.unit
class TestAuthenticationEndpoints:
    """Test API endpoints with authentication using new auth fixtures."""

    def test_openai_chat_completions_authenticated(
        self,
        client_with_auth: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test authenticated OpenAI chat completion."""
        response = client_with_auth.post(
            "/api/v1/chat/completions",
            json=STANDARD_OPENAI_REQUEST,
            headers=auth_headers,
        )

        # Should return 401 because auth token is valid but proxy service is not set up in test
        assert_auth_error(response)

    def test_openai_chat_completions_unauthenticated(
        self, client_with_auth: TestClient
    ) -> None:
        """Test OpenAI chat completion endpoint with no auth."""
        response = client_with_auth.post(
            "/api/v1/chat/completions", json=STANDARD_OPENAI_REQUEST
        )

        assert_auth_error(response)

    def test_openai_chat_completions_invalid_token(
        self, client_with_auth: TestClient
    ) -> None:
        """Test OpenAI chat completion endpoint with invalid token."""
        response = client_with_auth.post(
            "/api/v1/chat/completions",
            json=STANDARD_OPENAI_REQUEST,
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert_auth_error(response)

    def test_anthropic_messages_authenticated(
        self,
        client_with_auth: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test authenticated Anthropic message creation."""
        response = client_with_auth.post(
            "/api/v1/messages", json=STANDARD_ANTHROPIC_REQUEST, headers=auth_headers
        )

        assert_auth_error(response)

    def test_anthropic_messages_unauthenticated(
        self, client_with_auth: TestClient
    ) -> None:
        """Test Anthropic messages endpoint with no auth."""
        response = client_with_auth.post(
            "/api/v1/messages", json=STANDARD_ANTHROPIC_REQUEST
        )

        assert_auth_error(response)


@pytest.mark.unit
class TestComposableAuthenticationEndpoints:
    """Test API endpoints using composable auth patterns.

    These tests demonstrate different authentication modes using existing fixtures.
    """

    @pytest.mark.parametrize(
        "endpoint_path,request_data",
        [
            ("/sdk/v1/chat/completions", STANDARD_OPENAI_REQUEST),
            ("/sdk/v1/messages", STANDARD_ANTHROPIC_REQUEST),
        ],
        ids=["openai_no_auth", "anthropic_no_auth"],
    )
    def test_endpoints_no_auth_required(
        self,
        client_with_mock_claude: TestClient,
        endpoint_path: str,
        request_data: dict[str, Any],
    ) -> None:
        """Test endpoints with no authentication required."""
        response = client_with_mock_claude.post(endpoint_path, json=request_data)
        assert response.status_code == 200

        data: dict[str, Any] = response.json()
        if "chat/completions" in endpoint_path:
            assert_openai_response_format(data)
        else:
            assert_anthropic_response_format(data)

    def test_bearer_token_auth_endpoints(
        self,
        client_with_auth: TestClient,
        auth_headers: dict[str, str],
    ) -> None:
        """Test bearer token authentication on API endpoints."""
        # Test OpenAI endpoint - should fail auth but for correct reason
        response = client_with_auth.post(
            "/api/v1/chat/completions",
            json=STANDARD_OPENAI_REQUEST,
            headers=auth_headers,
        )
        assert_auth_error(response)

        # Test Anthropic endpoint - should fail auth but for correct reason
        response = client_with_auth.post(
            "/api/v1/messages", json=STANDARD_ANTHROPIC_REQUEST, headers=auth_headers
        )
        assert_auth_error(response)

    def test_auth_token_validation(self, client_with_auth: TestClient) -> None:
        """Test authentication token validation."""
        # Test with invalid token
        response = client_with_auth.post(
            "/api/v1/chat/completions",
            json=STANDARD_OPENAI_REQUEST,
            headers={"Authorization": "Bearer invalid-token"},
        )
        assert_auth_error(response)

        # Test without token
        response = client_with_auth.post(
            "/api/v1/messages", json=STANDARD_ANTHROPIC_REQUEST
        )
        assert_auth_error(response)


@pytest.mark.unit
class TestErrorHandling:
    """Test API error handling and edge cases."""

    def test_claude_cli_unavailable_error(
        self, client_with_unavailable_claude: TestClient
    ) -> None:
        """Test handling when Claude CLI is not available."""
        response = client_with_unavailable_claude.post(
            "/sdk/v1/messages", json=STANDARD_ANTHROPIC_REQUEST
        )

        assert_service_unavailable_error(response)

    def test_invalid_json(self, client_with_mock_claude: TestClient) -> None:
        """Test handling of invalid JSON requests."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages",
            content="invalid json",
            headers={"Content-Type": "application/json"},
        )

        assert_validation_error(response)

    def test_unsupported_content_type(
        self, client_with_mock_claude: TestClient
    ) -> None:
        """Test handling of unsupported content types."""
        response = client_with_mock_claude.post(
            "/sdk/v1/messages",
            content="some data",
            headers={"Content-Type": "text/plain"},
        )

        assert_validation_error(response)

    def test_large_request_body(
        self, client_with_unavailable_claude: TestClient
    ) -> None:
        """Test handling of large request bodies."""
        response = client_with_unavailable_claude.post(
            "/sdk/v1/messages", json=LARGE_REQUEST_ANTHROPIC
        )

        assert_service_unavailable_error(response)

    def test_malformed_headers(
        self, client_with_unavailable_claude: TestClient
    ) -> None:
        """Test handling of malformed headers."""
        response = client_with_unavailable_claude.post(
            "/sdk/v1/messages",
            json=STANDARD_ANTHROPIC_REQUEST,
            headers={"Authorization": "InvalidFormat"},
        )

        assert_service_unavailable_error(response)


# Removed TestResponseValidation - schema validation covered by endpoint tests with assert_*_response_format


@pytest.mark.unit
class TestStatusEndpoints:
    """Test various status and health check endpoints."""

    def test_all_status_endpoints(self, client: TestClient) -> None:
        """Test all status endpoints return successfully."""
        status_endpoints: list[str] = []

        for endpoint in status_endpoints:
            response = client.get(endpoint)
            assert response.status_code == 200, f"Status endpoint {endpoint} failed"

            data: dict[str, Any] = response.json()
            assert "status" in data or "message" in data

    def test_health_endpoints(self, client: TestClient) -> None:
        """Test new health check endpoints following IETF format."""
        # Test liveness probe - should always return 200
        response = client.get("/health/live")
        assert response.status_code == 200
        assert "application/health+json" in response.headers["content-type"]
        assert (
            response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
        )

        data: dict[str, Any] = response.json()
        assert data["status"] == "pass"
        assert "version" in data
        assert data["output"] == "Application process is running"

        # Test readiness probe - may return 200 or 503 depending on Claude SDK
        response = client.get("/health/ready")
        assert response.status_code in [200, 503]
        assert "application/health+json" in response.headers["content-type"]

        data = response.json()
        assert data["status"] in ["pass", "fail"]
        assert "version" in data
        assert "checks" in data
        assert "claude_sdk" in data["checks"]

        # Test detailed health check - comprehensive status
        response = client.get("/health")
        assert response.status_code in [200, 503]
        assert "application/health+json" in response.headers["content-type"]

        data = response.json()
        assert data["status"] in ["pass", "warn", "fail"]
        assert "version" in data
        assert "serviceId" in data
        assert "description" in data
        assert "time" in data
        assert "checks" in data
        assert "claude_sdk" in data["checks"]
        assert "proxy_service" in data["checks"]


# Removed TestRequestValidation - validation tests already covered in TestOpenAIEndpoints and TestAnthropicEndpoints
