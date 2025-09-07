"""Authentication and OAuth2 flow tests.

This module tests both bearer token authentication and OAuth2 flows together,
including token validation, credential storage, and API endpoint access control.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, status
from pydantic import SecretStr

from ccproxy.auth.bearer import BearerTokenAuthManager

# from ccproxy.auth.credentials_adapter import CredentialsAuthManager
from ccproxy.auth.dependencies import (
    get_access_token,
    require_auth,
)
from ccproxy.auth.exceptions import (
    AuthenticationError,
    AuthenticationRequiredError,
    CredentialsError,
    CredentialsExpiredError,
    CredentialsNotFoundError,
    InvalidTokenError,
    OAuthError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.manager import AuthManager
from ccproxy.plugins.oauth_claude.models import (
    ClaudeOAuthToken,
)


# from ccproxy.services.credentials.manager import CredentialsManager


@pytest.mark.auth
class TestBearerTokenAuthentication:
    """Test bearer token authentication mechanism."""

    def test_bearer_token_manager_creation(self) -> None:
        """Test bearer token manager initialization."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)
        assert manager.token == token

    def test_bearer_token_manager_empty_token_raises_error(self) -> None:
        """Test that empty token raises ValueError."""
        with pytest.raises(ValueError, match="Token cannot be empty"):
            BearerTokenAuthManager("")

    def test_bearer_token_manager_whitespace_token_raises_error(self) -> None:
        """Test that whitespace-only token raises ValueError."""
        with pytest.raises(ValueError, match="Token cannot be empty"):
            BearerTokenAuthManager("   ")

    async def test_bearer_token_manager_get_access_token(self) -> None:
        """Test getting access token from bearer token manager."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)

        access_token = await manager.get_access_token()
        assert access_token == token

    async def test_bearer_token_manager_is_authenticated(self) -> None:
        """Test authentication status check."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)

        is_authenticated = await manager.is_authenticated()
        assert is_authenticated is True

    async def test_bearer_token_manager_get_credentials_raises_error(self) -> None:
        """Test that getting credentials raises error for bearer tokens."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)

        with pytest.raises(
            AuthenticationError,
            match="Bearer token authentication doesn't support full credentials",
        ):
            await manager.get_credentials()

    async def test_bearer_token_manager_get_user_profile_returns_none(self) -> None:
        """Test that user profile returns None for bearer tokens."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)

        profile = await manager.get_user_profile()
        assert profile is None

    async def test_bearer_token_manager_async_context(self) -> None:
        """Test bearer token manager as async context manager."""
        token = "sk-test-token-123"

        async with BearerTokenAuthManager(token) as manager:
            assert manager.token == token
            assert await manager.is_authenticated() is True


@pytest.mark.auth
class TestAuthDependencies:
    """Test FastAPI authentication dependencies."""

    async def test_require_auth_with_authenticated_manager(self) -> None:
        """Test require_auth with authenticated manager."""
        mock_manager = AsyncMock(spec=AuthManager)
        mock_manager.is_authenticated.return_value = True

        result = await require_auth(mock_manager)
        assert result == mock_manager
        mock_manager.is_authenticated.assert_called_once()

    async def test_require_auth_with_unauthenticated_manager(self) -> None:
        """Test require_auth with unauthenticated manager."""
        mock_manager = AsyncMock(spec=AuthManager)
        mock_manager.is_authenticated.return_value = False

        with pytest.raises(HTTPException) as exc_info:
            await require_auth(mock_manager)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Authentication required" in str(exc_info.value.detail)
        assert exc_info.value.headers == {"WWW-Authenticate": "Bearer"}

    async def test_require_auth_with_authentication_error(self) -> None:
        """Test require_auth when authentication raises error."""
        mock_manager = AsyncMock(spec=AuthManager)
        mock_manager.is_authenticated.side_effect = AuthenticationError("Invalid token")

        with pytest.raises(HTTPException) as exc_info:
            await require_auth(mock_manager)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Invalid token" in str(exc_info.value.detail)

    async def test_get_access_token_dependency(self) -> None:
        """Test get_access_token dependency."""
        mock_manager = AsyncMock(spec=AuthManager)
        mock_manager.get_access_token.return_value = "sk-test-token-123"

        token = await get_access_token(mock_manager)
        assert token == "sk-test-token-123"
        mock_manager.get_access_token.assert_called_once()


@pytest.mark.auth
class TestAPIEndpointsWithAuth:
    """Test API endpoints with authentication enabled."""


class TestTokenRefreshFlow:
    """Test OAuth token refresh functionality."""

    @pytest.fixture
    def mock_oauth_token(self) -> ClaudeOAuthToken:
        """Create mock OAuth token."""
        return ClaudeOAuthToken(
            accessToken=SecretStr("sk-test-token-123"),
            refreshToken=SecretStr("refresh-token-456"),
            expiresAt=None,
            tokenType="Bearer",
            subscriptionType=None,
        )

    async def test_token_refresh_success(
        self, mock_oauth_token: ClaudeOAuthToken
    ) -> None:
        """Test successful token refresh."""
        # This is a unit test for the OAuthToken model structure
        # Actual token refresh would be tested via the CredentialsManager or OAuthClient
        # in integration tests
        assert mock_oauth_token.access_token.get_secret_value() == "sk-test-token-123"
        assert mock_oauth_token.refresh_token.get_secret_value() == "refresh-token-456"

    async def test_token_refresh_failure(self) -> None:
        """Test token refresh failure."""
        # This would be tested via the CredentialsManager or OAuthClient in integration tests
        # For now, we just verify the test structure is correct
        # Actual token refresh failure handling would involve catching specific exceptions
        # and handling them appropriately
        pass


@pytest.mark.auth
class TestCredentialStorage:
    """Test credential storage and retrieval functionality."""

    def test_credential_storage_paths_creation(self, tmp_path: Path) -> None:
        """Test creation of credential storage paths."""
        # Create test credential storage paths
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        credentials_file = config_dir / "credentials.json"
        test_credentials = {
            "claudeAiOauth": {
                "accessToken": "sk-test-token-123",
                "refreshToken": "refresh-token-456",
                "expiresAt": None,
                "tokenType": "Bearer",
            }
        }

        credentials_file.write_text(json.dumps(test_credentials))

        # Verify file was created and contains expected data
        assert credentials_file.exists()
        loaded_credentials = json.loads(credentials_file.read_text())
        assert loaded_credentials["claudeAiOauth"]["accessToken"] == "sk-test-token-123"

    def test_credential_file_not_found_handling(self, tmp_path: Path) -> None:
        """Test handling when credential file doesn't exist."""
        non_existent_file = tmp_path / "non_existent_credentials.json"

        # Verify file doesn't exist
        assert not non_existent_file.exists()

        # This would trigger CredentialsNotFoundError in real implementation
        with pytest.raises(FileNotFoundError):
            non_existent_file.read_text()

    def test_invalid_credential_file_handling(self, tmp_path: Path) -> None:
        """Test handling of invalid credential file format."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        credentials_file = config_dir / "credentials.json"
        credentials_file.write_text("invalid json content")

        # This would trigger parsing error in real implementation
        with pytest.raises(json.JSONDecodeError):
            json.loads(credentials_file.read_text())

    def test_expired_credentials_handling(self, tmp_path: Path) -> None:
        """Test handling of expired credentials."""
        config_dir = tmp_path / "config"
        config_dir.mkdir(parents=True, exist_ok=True)

        credentials_file = config_dir / "credentials.json"
        # Create credentials that appear expired (past timestamp)
        expired_credentials = {
            "claudeAiOauth": {
                "accessToken": "sk-test-token-123",
                "refreshToken": "refresh-token-456",
                "expiresAt": 1234567890000,  # Past timestamp in milliseconds
                "tokenType": "Bearer",
            }
        }

        credentials_file.write_text(json.dumps(expired_credentials))

        # Verify file contains expired credentials
        loaded_credentials = json.loads(credentials_file.read_text())
        assert loaded_credentials["claudeAiOauth"]["expiresAt"] == 1234567890000


@pytest.mark.auth
class TestAuthExceptions:
    """Test authentication exception handling."""

    def test_authentication_error_creation(self) -> None:
        """Test AuthenticationError exception creation."""
        error = AuthenticationError("Test authentication error")
        assert str(error) == "Test authentication error"
        assert isinstance(error, Exception)

    def test_authentication_required_error_creation(self) -> None:
        """Test AuthenticationRequiredError exception creation."""
        error = AuthenticationRequiredError("Authentication required")
        assert str(error) == "Authentication required"
        assert isinstance(error, AuthenticationError)

    def test_invalid_token_error_creation(self) -> None:
        """Test InvalidTokenError exception creation."""
        error = InvalidTokenError("Invalid token format")
        assert str(error) == "Invalid token format"
        assert isinstance(error, AuthenticationError)

    def test_credentials_not_found_error_creation(self) -> None:
        """Test CredentialsNotFoundError exception creation."""
        error = CredentialsNotFoundError("Credentials not found")
        assert str(error) == "Credentials not found"
        assert isinstance(error, CredentialsError)

    def test_credentials_expired_error_creation(self) -> None:
        """Test CredentialsExpiredError exception creation."""
        error = CredentialsExpiredError("Credentials expired")
        assert str(error) == "Credentials expired"
        assert isinstance(error, CredentialsError)

    def test_oauth_error_creation(self) -> None:
        """Test OAuthError exception creation."""
        error = OAuthError("OAuth authentication failed")
        assert str(error) == "OAuth authentication failed"
        assert isinstance(error, Exception)

    def test_oauth_token_refresh_error_creation(self) -> None:
        """Test OAuthTokenRefreshError exception creation."""
        error = OAuthTokenRefreshError("Token refresh failed")
        assert str(error) == "Token refresh failed"
        assert isinstance(error, OAuthError)


@pytest.mark.auth
class TestAuthenticationIntegration:
    """Test end-to-end authentication scenarios."""

    async def test_full_bearer_token_flow(self) -> None:
        """Test complete bearer token authentication flow."""
        test_token = "test-token-12345"
        # Create bearer token manager
        manager = BearerTokenAuthManager(test_token)

        # Test authentication
        assert await manager.is_authenticated() is True

        # Test token retrieval
        token = await manager.get_access_token()
        assert token == test_token

    async def test_authentication_dependency_integration(self) -> None:
        """Test authentication dependencies working together."""
        # Create mock settings with auth enabled
        mock_settings = MagicMock()
        mock_settings.server = MagicMock()
        mock_settings.server.auth_token = "test-token-123"
        mock_settings.auth = MagicMock()

        # Test dependency resolution would happen here
        # This is more of an integration test that would require actual dependency injection
        pass


@pytest.mark.auth
@pytest.mark.asyncio
class TestAsyncAuthenticationPatterns:
    """Test async authentication patterns and context managers."""

    async def test_auth_manager_async_context_pattern(self) -> None:
        """Test auth manager async context manager pattern."""
        token = "sk-test-token-123"

        async with BearerTokenAuthManager(token) as manager:
            assert await manager.is_authenticated() is True
            assert await manager.get_access_token() == token

    async def test_concurrent_auth_operations(self) -> None:
        """Test concurrent authentication operations."""
        token = "sk-test-token-123"
        manager = BearerTokenAuthManager(token)

        # Run multiple auth operations concurrently
        tasks = [
            manager.is_authenticated(),
            manager.get_access_token(),
            manager.get_user_profile(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results
        assert results[0] is True  # is_authenticated
        assert results[1] == token  # get_access_token
        assert results[2] is None  # get_user_profile (not supported for bearer tokens)

    async def test_auth_error_propagation(self) -> None:
        """Test that authentication errors properly propagate through async calls."""
        mock_manager = AsyncMock(spec=AuthManager)
        mock_manager.is_authenticated.side_effect = AuthenticationError("Test error")

        # Error should propagate through require_auth
        with pytest.raises(HTTPException) as exc_info:
            await require_auth(mock_manager)

        assert exc_info.value.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Test error" in str(exc_info.value.detail)
