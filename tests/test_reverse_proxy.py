"""Tests for reverse proxy functionality."""

import json
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

from ccproxy.services.request_transformer import RequestTransformer


class TestRequestTransformer:
    """Test request transformer functionality."""

    def test_transform_system_prompt_string(self):
        """Test system prompt transformation with string input."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "system": "You are a helpful assistant.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result = transformer.transform_system_prompt(body)
        data = json.loads(result.decode("utf-8"))

        # Should be converted to array with Claude Code first
        assert isinstance(data["system"], list)
        assert len(data["system"]) == 2
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"
        assert data["system"][1]["text"] == "You are a helpful assistant."

    def test_transform_system_prompt_array(self):
        """Test system prompt transformation with array input."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "system": [
                    {"type": "text", "text": "You are helpful."},
                    {"type": "text", "text": "Be concise."},
                ],
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result = transformer.transform_system_prompt(body)
        data = json.loads(result.decode("utf-8"))

        # Should prepend Claude Code prompt
        assert isinstance(data["system"], list)
        assert len(data["system"]) == 3
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"
        assert data["system"][1]["text"] == "You are helpful."
        assert data["system"][2]["text"] == "Be concise."

    def test_transform_system_prompt_already_correct(self):
        """Test that correct system prompt is converted to proper array format."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "system": "You are Claude Code, Anthropic's official CLI for Claude.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result = transformer.transform_system_prompt(body)
        data = json.loads(result.decode("utf-8"))

        # Should be converted to proper array format with cache_control
        assert isinstance(data["system"], list)
        assert len(data["system"]) == 1
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_transform_system_prompt_no_system(self):
        """Test system prompt injection when no system prompt exists."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result = transformer.transform_system_prompt(body)
        data = json.loads(result.decode("utf-8"))

        # Should inject Claude Code prompt as array
        assert isinstance(data["system"], list)
        assert len(data["system"]) == 1
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_transform_request_body_messages_endpoint(self):
        """Test full request body transformation for messages endpoint."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "system": "You are helpful.",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result = transformer.transform_request_body(body, "/v1/messages")
        data = json.loads(result.decode("utf-8"))

        # Should preserve original model
        assert data["model"] == "claude-3-5-sonnet-latest"

        # Should transform system prompt
        assert isinstance(data["system"], list)
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )

    def test_transform_request_body_non_messages_endpoint(self):
        """Test that non-messages endpoints are not transformed."""
        transformer = RequestTransformer()

        body = json.dumps(
            {"model": "claude-3-5-sonnet-latest", "prompt": "Hello"}
        ).encode("utf-8")

        result = transformer.transform_request_body(body, "/v1/completions")

        # Should be unchanged
        assert result == body

    def test_create_proxy_headers(self):
        """Test proxy header creation."""
        transformer = RequestTransformer()

        original_headers = {
            "content-type": "application/json",
            "accept": "text/event-stream",
            "connection": "keep-alive",
            "authorization": "Bearer old-token",  # Should be replaced
        }

        headers = transformer.create_proxy_headers(original_headers, "new-token")

        # Should have OAuth headers
        assert headers["Authorization"] == "Bearer new-token"
        assert "claude-code-20250219,oauth-2025-04-20" in headers["anthropic-beta"]
        assert headers["anthropic-version"] == "2023-06-01"

        # Should have Claude CLI headers
        assert headers["x-app"] == "cli"
        assert headers["User-Agent"] == "claude-cli/1.0.43 (external, cli)"
        assert headers["anthropic-dangerous-direct-browser-access"] == "true"

        # Should have Stainless headers
        assert headers["X-Stainless-Lang"] == "js"
        assert headers["X-Stainless-Retry-Count"] == "0"
        assert headers["X-Stainless-Timeout"] == "60"

        # Should preserve essential headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "text/event-stream"
        assert headers["Connection"] == "keep-alive"

    def test_create_proxy_headers_defaults(self):
        """Test proxy header creation with defaults."""
        transformer = RequestTransformer()

        headers = transformer.create_proxy_headers({}, "token")

        # Should have defaults
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"
        assert headers["Connection"] == "keep-alive"

        # Should have Claude CLI headers by default
        assert headers["x-app"] == "cli"
        assert headers["User-Agent"] == "claude-cli/1.0.43 (external, cli)"

    def test_create_proxy_headers_strips_user_agent(self):
        """Test that original user-agent headers are replaced with Claude CLI user-agent."""
        transformer = RequestTransformer()

        original_headers = {
            "content-type": "application/json",
            "user-agent": "MyApp/1.0.0",  # Should be ignored/replaced
            "User-Agent": "AnotherApp/2.0.0",  # Should also be ignored/replaced
            "accept": "application/json",
            "authorization": "Bearer old-token",  # Should be replaced
            "host": "example.com",  # Should be stripped
            "x-forwarded-for": "127.0.0.1",  # Should be stripped
        }

        headers = transformer.create_proxy_headers(original_headers, "new-token")

        # Should NOT contain problematic headers (but User-Agent is now set to Claude CLI)
        assert "host" not in headers
        assert "x-forwarded-for" not in headers

        # Should have Claude CLI User-Agent (not the original ones)
        assert headers["User-Agent"] == "claude-cli/1.0.43 (external, cli)"

        # Should have required OAuth headers
        assert headers["Authorization"] == "Bearer new-token"
        assert "oauth-2025-04-20" in headers["anthropic-beta"]
        assert headers["anthropic-version"] == "2023-06-01"

        # Should preserve essential headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

    def test_transform_path_removes_openai_prefix(self):
        """Test that /openai prefix is removed and endpoints are converted."""
        transformer = RequestTransformer()

        # Test OpenAI path transformation with endpoint conversion
        assert (
            transformer.transform_path("/openai/v1/chat/completions") == "/v1/messages"
        )
        assert transformer.transform_path("/v1/chat/completions") == "/v1/messages"
        assert transformer.transform_path("/openai/v1/models") == "/v1/models"

        # Test non-OpenAI paths remain unchanged
        assert transformer.transform_path("/v1/messages") == "/v1/messages"
        assert transformer.transform_path("/health") == "/health"

    def test_is_openai_request_detection(self):
        """Test OpenAI request detection."""
        transformer = RequestTransformer()

        # Test path-based detection
        assert transformer._is_openai_request("/openai/v1/chat/completions", b"")
        assert transformer._is_openai_request("/v1/chat/completions", b"")

        # Test body-based detection
        openai_body = json.dumps(
            {"model": "gpt-4", "messages": [{"role": "user", "content": "Hello"}]}
        ).encode("utf-8")
        assert transformer._is_openai_request("/v1/chat/completions", openai_body)

        # Test non-OpenAI request
        anthropic_body = json.dumps(
            {
                "model": "claude-3-5-sonnet-latest",
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")
        assert not transformer._is_openai_request("/v1/messages", anthropic_body)

    def test_transform_openai_to_anthropic(self):
        """Test OpenAI to Anthropic format transformation."""
        transformer = RequestTransformer()

        openai_body = json.dumps(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                ],
                "temperature": 0.7,
            }
        ).encode("utf-8")

        result = transformer._transform_openai_to_anthropic(openai_body)
        data = json.loads(result.decode("utf-8"))

        # Should preserve original OpenAI model after conversion
        assert (
            data["model"] == "claude-3-7-sonnet-20250219"
        )  # gpt-4o maps to this via OpenAI translator

        # Should have system prompt with Claude Code first
        assert isinstance(data["system"], list)
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data["system"][0]["cache_control"]["type"] == "ephemeral"
        assert data["system"][1]["text"] == "You are helpful"

        # Should preserve messages and other parameters
        assert len(data["messages"]) == 1
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["temperature"] == 0.7

    def test_transform_system_prompt_edge_cases(self):
        """Test edge cases for system prompt transformation."""
        transformer = RequestTransformer()

        # Test with empty string system prompt
        body1 = json.dumps(
            {
                "model": "claude-3-5-sonnet",
                "system": "",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result1 = transformer.transform_system_prompt(body1)
        data1 = json.loads(result1.decode("utf-8"))

        assert isinstance(data1["system"], list)
        assert len(data1["system"]) == 2
        assert (
            data1["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data1["system"][0]["cache_control"]["type"] == "ephemeral"
        assert data1["system"][1]["text"] == ""

        # Test with empty array system prompt
        body2 = json.dumps(
            {
                "model": "claude-3-5-sonnet",
                "system": [],
                "messages": [{"role": "user", "content": "Hello"}],
            }
        ).encode("utf-8")

        result2 = transformer.transform_system_prompt(body2)
        data2 = json.loads(result2.decode("utf-8"))

        assert isinstance(data2["system"], list)
        assert len(data2["system"]) == 1
        assert (
            data2["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert data2["system"][0]["cache_control"]["type"] == "ephemeral"

    def test_transform_system_prompt_invalid_json(self):
        """Test handling of invalid JSON input."""
        transformer = RequestTransformer()

        # Test with invalid JSON
        invalid_body = b"invalid json content"
        result = transformer.transform_system_prompt(invalid_body)

        # Should return original body unchanged
        assert result == invalid_body

        # Test with non-UTF8 content
        non_utf8_body = b"\xff\xfe\x00\x00"
        result2 = transformer.transform_system_prompt(non_utf8_body)

        # Should return original body unchanged
        assert result2 == non_utf8_body

    def test_get_claude_code_prompt_helper(self):
        """Test the helper function for Claude Code prompt."""
        from ccproxy.services.request_transformer import (
            get_claude_code_prompt,
        )

        prompt = get_claude_code_prompt()

        assert isinstance(prompt, dict)
        assert prompt["type"] == "text"
        assert (
            prompt["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )
        assert prompt["cache_control"]["type"] == "ephemeral"

    def test_transform_system_prompt_preserves_other_fields(self):
        """Test that transformation preserves all other request fields."""
        transformer = RequestTransformer()

        body = json.dumps(
            {
                "model": "claude-3-5-sonnet",
                "system": "You are helpful",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 100,
                "temperature": 0.5,
                "stream": True,
                "custom_field": "custom_value",
            }
        ).encode("utf-8")

        result = transformer.transform_system_prompt(body)
        data = json.loads(result.decode("utf-8"))

        # Should preserve all other fields
        assert data["model"] == "claude-3-5-sonnet"
        assert data["max_tokens"] == 100
        assert data["temperature"] == 0.5
        assert data["stream"] is True
        assert data["custom_field"] == "custom_value"

        # Should transform system prompt
        assert isinstance(data["system"], list)
        assert len(data["system"]) == 2
        assert (
            data["system"][0]["text"]
            == "You are Claude Code, Anthropic's official CLI for Claude."
        )

    def test_create_proxy_headers_strips_api_keys(self):
        """Test that API keys are properly stripped from headers."""
        transformer = RequestTransformer()

        original_headers = {
            "content-type": "application/json",
            "authorization": "Bearer client-api-key-123",  # Should be stripped
            "x-api-key": "sk-ant-client-key-456",  # Should be stripped
            "X-API-KEY": "another-client-key",  # Case variant - should be stripped
            "Authorization": "Bearer another-token",  # Case variant - should be stripped
            "accept": "application/json",
            "user-agent": "MyClient/1.0",
        }

        headers = transformer.create_proxy_headers(original_headers, "oauth-token-789")

        # Should have OAuth token (not client keys)
        assert headers["Authorization"] == "Bearer oauth-token-789"

        # Should NOT contain any client authentication headers
        assert "x-api-key" not in headers
        assert "X-API-KEY" not in headers
        assert "X-Api-Key" not in headers

        # Should not contain the old authorization values
        assert "client-api-key-123" not in str(headers.values())
        assert "sk-ant-client-key-456" not in str(headers.values())
        assert "another-client-key" not in str(headers.values())
        assert "another-token" not in str(headers.values())

        # Should preserve safe headers
        assert headers["Content-Type"] == "application/json"
        assert headers["Accept"] == "application/json"

        # Should have Claude CLI headers
        assert headers["User-Agent"] == "claude-cli/1.0.43 (external, cli)"
        assert headers["x-app"] == "cli"

    def test_create_proxy_headers_no_auth_leakage(self):
        """Test comprehensive check that no authentication data leaks through."""
        transformer = RequestTransformer()

        # Test with various auth header formats
        original_headers = {
            "authorization": "Bearer sk-ant-api03-secret-key",
            "x-api-key": "sk-ant-another-secret",
            "X-API-KEY": "CAPS-SECRET-KEY",
            "Authorization": "Basic dXNlcjpwYXNz",  # Base64 user:pass
            "proxy-authorization": "Bearer proxy-secret",
            "www-authenticate": "Bearer realm=api",
            "content-type": "application/json",
        }

        headers = transformer.create_proxy_headers(original_headers, "safe-oauth-token")

        # Convert all header values to a single string for comprehensive check
        all_header_values = " ".join(str(v) for v in headers.values()).lower()

        # Ensure no secrets leak through
        assert "sk-ant-api03-secret-key" not in all_header_values
        assert "sk-ant-another-secret" not in all_header_values
        assert "caps-secret-key" not in all_header_values
        assert "dxnlcjpwyxnz" not in all_header_values  # Base64 decoded check
        assert "proxy-secret" not in all_header_values

        # Should only have the safe OAuth token
        assert headers["Authorization"] == "Bearer safe-oauth-token"
        assert "safe-oauth-token" in all_header_values


class TestReverseProxyAuthentication:
    """Test authentication for reverse proxy endpoints."""

    @pytest.fixture
    def app_with_auth(self, monkeypatch):
        """Create app with authentication enabled."""
        from ccproxy.config.settings import Settings
        from ccproxy.main import create_app

        # Create settings with authentication enabled
        auth_settings = Settings(
            auth_token="test-auth-token-123",
            reverse_proxy_target_url="https://api.anthropic.com",
            reverse_proxy_timeout=30.0,
            _env_file=None,  # Don't load from file
        )

        # Mock the global get_settings function to return our auth settings
        # This is necessary because the auth middleware uses get_settings()
        # instead of the settings passed to create_app()
        monkeypatch.setattr(
            "ccproxy.middleware.auth.get_settings", lambda: auth_settings
        )

        # Create app with auth settings
        return create_app(auth_settings)

    @pytest.fixture
    def app_no_auth(self, monkeypatch):
        """Create app with authentication disabled."""
        from ccproxy.config.settings import Settings
        from ccproxy.main import create_app

        # Create settings with authentication disabled
        no_auth_settings = Settings(
            auth_token=None,
            reverse_proxy_target_url="https://api.anthropic.com",
            reverse_proxy_timeout=30.0,
            _env_file=None,  # Don't load from file
        )

        # Mock the global get_settings function to return our no-auth settings
        monkeypatch.setattr(
            "ccproxy.middleware.auth.get_settings", lambda: no_auth_settings
        )

        # Create app with no auth settings
        return create_app(no_auth_settings)

    def test_reverse_proxy_requires_auth_when_configured(
        self, app_with_auth, monkeypatch
    ):
        """Test that reverse proxy requires authentication when auth is configured."""
        client = TestClient(app_with_auth)

        # Request without authentication should fail at middleware level
        response = client.post(
            "/unclaude/v1/messages", json={"model": "claude-3-5-sonnet", "messages": []}
        )

        assert response.status_code == 401
        assert "authentication_error" in response.json()["detail"]["error"]["type"]
        assert (
            "Missing authentication token"
            in response.json()["detail"]["error"]["message"]
        )

    def test_reverse_proxy_accepts_valid_auth_x_api_key(
        self, app_with_auth, monkeypatch
    ):
        """Test that reverse proxy accepts valid x-api-key authentication."""

        # Create a mock credentials file in test location
        import json
        from pathlib import Path

        test_creds_dir = Path("/tmp/ccproxy-test/.claude")
        test_creds_dir.mkdir(parents=True, exist_ok=True)
        test_creds_file = test_creds_dir / ".credentials.json"

        # Create valid test credentials
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        test_creds = {
            "claudeAiOauth": {
                "accessToken": "oauth-token-123",
                "refreshToken": "test-refresh-token",
                "expiresAt": future_ms,
                "scopes": ["user:inference"],
                "subscriptionType": "test",
            }
        }

        test_creds_file.write_text(json.dumps(test_creds))

        # Mock httpx to avoid actual API calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"message": "success"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.reason_phrase = "OK"

        async def mock_request(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

        client = TestClient(app_with_auth)

        # Request with valid x-api-key should succeed
        response = client.post(
            "/unclaude/v1/messages",
            json={"model": "claude-3-5-sonnet", "messages": []},
            headers={"x-api-key": "test-auth-token-123"},
        )

        assert response.status_code == 200

    def test_reverse_proxy_accepts_valid_auth_bearer(self, app_with_auth, monkeypatch):
        """Test that reverse proxy accepts valid Authorization Bearer authentication."""

        # Create a mock credentials file in test location
        import json
        from pathlib import Path

        test_creds_dir = Path("/tmp/ccproxy-test/.claude")
        test_creds_dir.mkdir(parents=True, exist_ok=True)
        test_creds_file = test_creds_dir / ".credentials.json"

        # Create valid test credentials
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        test_creds = {
            "claudeAiOauth": {
                "accessToken": "oauth-token-123",
                "refreshToken": "test-refresh-token",
                "expiresAt": future_ms,
                "scopes": ["user:inference"],
                "subscriptionType": "test",
            }
        }

        test_creds_file.write_text(json.dumps(test_creds))

        # Mock httpx to avoid actual API calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"message": "success"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.reason_phrase = "OK"

        async def mock_request(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

        client = TestClient(app_with_auth)

        # Request with valid Authorization Bearer should succeed
        response = client.post(
            "/unclaude/v1/messages",
            json={"model": "claude-3-5-sonnet", "messages": []},
            headers={"Authorization": "Bearer test-auth-token-123"},
        )

        assert response.status_code == 200

    def test_reverse_proxy_rejects_invalid_auth(self, app_with_auth, monkeypatch):
        """Test that reverse proxy rejects invalid authentication."""
        client = TestClient(app_with_auth)

        # Request with invalid x-api-key should fail at middleware level
        response = client.post(
            "/unclaude/v1/messages",
            json={"model": "claude-3-5-sonnet", "messages": []},
            headers={"x-api-key": "invalid-token"},
        )

        assert response.status_code == 401
        assert "authentication_error" in response.json()["detail"]["error"]["type"]
        assert (
            "Invalid authentication token"
            in response.json()["detail"]["error"]["message"]
        )

        # Request with invalid Authorization Bearer should fail at middleware level
        response2 = client.post(
            "/unclaude/v1/messages",
            json={"model": "claude-3-5-sonnet", "messages": []},
            headers={"Authorization": "Bearer invalid-token"},
        )

        assert response2.status_code == 401
        assert "authentication_error" in response2.json()["detail"]["error"]["type"]

    def test_reverse_proxy_no_auth_required_when_disabled(
        self, app_no_auth, monkeypatch
    ):
        """Test that reverse proxy doesn't require auth when authentication is disabled."""

        # Create a mock credentials file in test location
        import json
        from pathlib import Path

        test_creds_dir = Path("/tmp/ccproxy-test/.claude")
        test_creds_dir.mkdir(parents=True, exist_ok=True)
        test_creds_file = test_creds_dir / ".credentials.json"

        # Create valid test credentials
        from datetime import UTC, datetime, timedelta

        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        test_creds = {
            "claudeAiOauth": {
                "accessToken": "oauth-token-123",
                "refreshToken": "test-refresh-token",
                "expiresAt": future_ms,
                "scopes": ["user:inference"],
                "subscriptionType": "test",
            }
        }

        test_creds_file.write_text(json.dumps(test_creds))

        # Mock httpx to avoid actual API calls
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'{"message": "success"}'
        mock_response.headers = {"content-type": "application/json"}
        mock_response.reason_phrase = "OK"

        async def mock_request(*args, **kwargs):
            return mock_response

        monkeypatch.setattr("httpx.AsyncClient.request", mock_request)

        client = TestClient(app_no_auth)

        # Request without authentication should succeed when auth is disabled
        response = client.post(
            "/unclaude/v1/messages", json={"model": "claude-3-5-sonnet", "messages": []}
        )

        assert response.status_code == 200
