"""Tests for authentication middleware."""

from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from claude_code_proxy.config.settings import get_settings
from claude_code_proxy.middleware.auth import verify_token


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

        # Should raise HTTPException for missing credentials
        with pytest.raises(HTTPException) as exc_info:
            verify_token(None, mock_request)

        assert exc_info.value.status_code == 401
        assert "authentication_error" in str(exc_info.value.detail)
        assert "Missing authentication token" in str(exc_info.value.detail)

    def test_verify_token_invalid_token(self, monkeypatch):
        """Test that invalid token raises authentication error."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"

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

    def test_verify_token_valid_token(self, monkeypatch):
        """Test that valid token passes authentication."""
        # Mock settings with auth token
        mock_settings = Mock()
        mock_settings.auth_token = "correct-token-123"
        monkeypatch.setattr(
            "claude_code_proxy.middleware.auth.get_settings", lambda: mock_settings
        )

        mock_request = Mock()
        mock_request.url.path = "/v1/chat/completions"

        # Mock valid credentials
        valid_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="correct-token-123"
        )

        # Should not raise any exception
        verify_token(valid_credentials, mock_request)
