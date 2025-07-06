"""Tests for authentication middleware."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.middleware.auth import extract_token_from_headers, verify_token


class TestAuthentication:
    """Test authentication middleware functionality."""

    def test_verify_token_no_auth_configured(self, monkeypatch):
        """Test that authentication is skipped when no token is configured."""
        # Mock settings with no auth token
        mock_settings = Mock()
        mock_settings.auth_token = None
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {}

        # Should not raise any exception
        verify_token(None, mock_request)

    def test_verify_token_missing_credentials(self, monkeypatch):
        """Test that missing credentials raise authentication error."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "test-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {}

        # Should raise HTTPException for missing credentials
        with pytest.raises(HTTPException) as exc_info:
            verify_token(None, mock_request)

        assert exc_info.value.status_code == 401
        assert "authentication_error" in str(exc_info.value.detail)
        assert "Missing authentication token" in str(exc_info.value.detail)

    def test_verify_token_invalid_bearer_token(self, monkeypatch):
        """Test that invalid Bearer token raises authentication error."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {}

        # Mock invalid credentials
        invalid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="wrong-token-456"
        )

        # Should raise HTTPException for invalid token
        with pytest.raises(HTTPException) as exc_info:
            verify_token(invalid_credentials, mock_request)

        assert exc_info.value.status_code == 401
        assert "authentication_error" in str(exc_info.value.detail)
        assert "Invalid authentication token" in str(exc_info.value.detail)

    def test_verify_token_valid_bearer_token(self, monkeypatch):
        """Test that valid Bearer token passes authentication."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"
        mock_request.headers = {}

        # Mock valid credentials
        valid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="correct-token-123"
        )

        # Should not raise any exception
        verify_token(valid_credentials, mock_request)

    def test_verify_token_valid_x_api_key(self, monkeypatch):
        """Test that valid x-api-key header passes authentication."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/messages"
        mock_request.headers = {"x-api-key": "correct-token-123"}

        # Should not raise any exception (no Bearer credentials)
        verify_token(None, mock_request)

    def test_verify_token_invalid_x_api_key(self, monkeypatch):
        """Test that invalid x-api-key header raises authentication error."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/messages"
        mock_request.headers = {"x-api-key": "wrong-token-456"}

        # Should raise HTTPException for invalid token
        with pytest.raises(HTTPException) as exc_info:
            verify_token(None, mock_request)

        assert exc_info.value.status_code == 401
        assert "authentication_error" in str(exc_info.value.detail)
        assert "Invalid authentication token" in str(exc_info.value.detail)

    def test_x_api_key_takes_precedence(self, monkeypatch):
        """Test that x-api-key header takes precedence over Bearer token."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/messages"
        # x-api-key has correct token, Bearer has wrong token
        mock_request.headers = {"x-api-key": "correct-token-123"}

        # Mock Bearer credentials with wrong token
        bearer_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="wrong-token-456"
        )

        # Should not raise any exception (x-api-key takes precedence)
        verify_token(bearer_credentials, mock_request)

    def test_extract_token_from_headers_x_api_key(self):
        """Test extracting token from x-api-key header."""
        mock_request = Mock()
        mock_request.headers = {"x-api-key": "test-token-123"}

        token = extract_token_from_headers(None, mock_request)
        assert token == "test-token-123"

    def test_extract_token_from_headers_bearer(self):
        """Test extracting token from Bearer header."""
        mock_request = Mock()
        mock_request.headers = {}

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="test-token-456"
        )

        token = extract_token_from_headers(credentials, mock_request)
        assert token == "test-token-456"

    def test_extract_token_from_headers_none(self):
        """Test extracting token when no headers are present."""
        mock_request = Mock()
        mock_request.headers = {}

        token = extract_token_from_headers(None, mock_request)
        assert token is None

    def test_empty_token_values(self, monkeypatch):
        """Test that empty token values are treated as missing tokens."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/messages"
        # Empty x-api-key value (empty string is falsy, so treated as no token)
        mock_request.headers = {"x-api-key": ""}

        # Should raise HTTPException for missing token
        with pytest.raises(HTTPException) as exc_info:
            verify_token(None, mock_request)

        assert exc_info.value.status_code == 401
        assert "authentication_error" in str(exc_info.value.detail)
        # Empty string is falsy, so it's treated as missing token
        assert "Missing authentication token" in str(exc_info.value.detail)

    def test_both_headers_with_same_token(self, monkeypatch):
        """Test that authentication works when both headers have the same correct token."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/messages"
        # Both headers have the same correct token
        mock_request.headers = {"x-api-key": "correct-token-123"}

        # Mock Bearer credentials with same token
        bearer_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="correct-token-123"
        )

        # Should not raise any exception
        verify_token(bearer_credentials, mock_request)
