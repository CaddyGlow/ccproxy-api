"""Simple verification that new URL structure works correctly."""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from ccproxy.services.credentials import CredentialsManager


@pytest.mark.integration
class TestNewURLStructure:
    """Test that the new URL structure works correctly."""

    def test_health_endpoint(self, test_client: TestClient):
        """Test health check endpoint."""
        response = test_client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    @patch("ccproxy.routers.anthropic.ClaudeClient")
    def test_new_claude_code_path(
        self, mock_client_class, test_client: TestClient, sample_claude_response
    ):
        """Test new /cc/v1/* path works."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = sample_claude_response
        mock_client_class.return_value = mock_client

        response = test_client.post(
            "/cc/v1/messages",
            json={
                "model": "claude-3-5-sonnet-20241022",
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 100,
            },
        )

        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")

        assert response.status_code == 200
        assert response.json()["type"] == "message"

    @patch("ccproxy.routers.openai.ClaudeClient")
    def test_new_openai_path(
        self, mock_client_class, test_client: TestClient, sample_claude_response
    ):
        """Test new /cc/openai/v1/* path works."""
        mock_client = AsyncMock()
        mock_client.create_completion.return_value = sample_claude_response
        mock_client_class.return_value = mock_client

        response = test_client.post(
            "/cc/openai/v1/chat/completions",
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        assert response.json()["object"] == "chat.completion"

    @patch("httpx.AsyncClient.request")
    async def test_minimal_proxy_mode(self, mock_request, test_client: TestClient):
        """Test /min/* proxy with minimal transformations."""
        # Mock the credentials manager's get_access_token to return our test token
        with patch.object(
            CredentialsManager,
            "get_access_token",
            AsyncMock(return_value="test-oauth-token"),
        ):
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = json.dumps(
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            ).encode()
            mock_request.return_value = mock_response

            response = test_client.post(
                "/min/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert response.status_code == 200

            # Verify minimal headers
            assert mock_request.called
            headers = mock_request.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-oauth-token"
            assert "x-app" not in headers

    @patch("httpx.AsyncClient.request")
    async def test_full_proxy_mode(self, mock_request, test_client: TestClient):
        """Test /full/* proxy with full transformations."""
        # Mock the credentials manager's get_access_token to return our test token
        with patch.object(
            CredentialsManager,
            "get_access_token",
            AsyncMock(return_value="test-oauth-token"),
        ):
            mock_response = AsyncMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "application/json"}
            mock_response.content = json.dumps(
                {
                    "id": "msg_123",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            ).encode()
            mock_request.return_value = mock_response

            response = test_client.post(
                "/full/v1/messages",
                json={
                    "model": "claude-3-5-sonnet-20241022",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )

            assert response.status_code == 200

            # Verify full headers
            headers = mock_request.call_args.kwargs["headers"]
            assert headers["Authorization"] == "Bearer test-oauth-token"
            assert headers["x-app"] == "cli"

    def test_legacy_paths_exist(self, test_client: TestClient):
        """Test legacy paths exist for backward compatibility."""
        # Just verify the endpoints exist and are reachable
        # We've already tested the functionality in other tests

        # Test that legacy paths don't return 404
        legacy_endpoints = [
            (
                "/v1/chat/completions",
                {"model": "claude-3-5-sonnet-20241022", "messages": []},
            ),
            ("/openai/v1/chat/completions", {"model": "gpt-4", "messages": []}),
        ]

        for path, minimal_data in legacy_endpoints:
            response = test_client.post(path, json=minimal_data)
            # Should not be 404 (endpoint not found)
            # May be 422 (validation) or 401 (auth) or other errors, but not 404
            assert response.status_code != 404, f"Legacy endpoint {path} returned 404"
