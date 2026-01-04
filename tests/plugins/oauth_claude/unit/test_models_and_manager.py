"""OAuth Claude plugin model and manager tests moved from core tests.

Covers:
- ClaudeTokenWrapper/ClaudeProfileInfo parsing and properties
- ClaudeApiTokenManager with GenericJsonStorage
- BaseTokenManager.get_unified_profile using Claude profile
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from ccproxy.auth.exceptions import OAuthTokenRefreshError
from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.storage.generic import GenericJsonStorage
from ccproxy.plugins.oauth_claude.models import (
    ClaudeCredentials,
    ClaudeOAuthToken,
    ClaudeProfileInfo,
    ClaudeTokenWrapper,
)


class TestClaudeModels:
    """Test Claude-specific models."""

    def test_claude_token_wrapper(self, tmp_path: object) -> None:
        """Test ClaudeTokenWrapper functionality."""
        # Create test credentials
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_access"),
            refreshToken=SecretStr("test_refresh"),
            expiresAt=int(datetime.now(UTC).timestamp() * 1000) + 3600000,  # 1 hour
            scopes=["read", "write"],
            subscriptionType="pro",
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)

        # Create wrapper with home path isolated from user environment
        with patch("pathlib.Path.home", return_value=tmp_path):
            wrapper = ClaudeTokenWrapper(credentials=credentials)

            # Test properties (these are Pydantic computed_field and @property)
            access_val: str = wrapper.access_token_value  # type: ignore[assignment]
            refresh_val: str | None = wrapper.refresh_token_value  # type: ignore[assignment]
            expired: bool = wrapper.is_expired  # type: ignore[assignment]

            assert access_val == "test_access"
            assert refresh_val == "test_refresh"
            assert expired is False
            # subscription_type comes from the local profile or fallback to "pro"
            # In this test context, it will likely be the fallback value
            assert wrapper.scopes == ["read", "write"]

    def test_claude_token_wrapper_expired(self) -> None:
        """Test ClaudeTokenWrapper with expired token."""
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_access"),
            refreshToken=SecretStr("test_refresh"),
            expiresAt=int(datetime.now(UTC).timestamp() * 1000) - 3600000,  # 1 hour ago
            subscriptionType=None,
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)
        wrapper = ClaudeTokenWrapper(credentials=credentials)

        expired: bool = wrapper.is_expired  # type: ignore[assignment]
        assert expired is True

    def test_claude_profile_from_api_response(self) -> None:
        """Test creating ClaudeProfileInfo from API response."""
        api_response = {
            "account": {
                "uuid": "test-uuid",
                "email": "user@example.com",
                "full_name": "Test User",
                "has_claude_pro": True,
                "has_claude_max": False,
            },
            "organization": {"uuid": "org-uuid", "name": "Test Org"},
        }

        profile = ClaudeProfileInfo.from_api_response(api_response)

        assert profile.account_id == "test-uuid"
        assert profile.email == "user@example.com"
        assert profile.display_name == "Test User"
        assert profile.provider_type == "claude-api"
        assert profile.has_claude_pro is True
        assert profile.has_claude_max is False
        assert profile.organization_name == "Test Org"
        assert profile.extras == api_response  # Full response preserved


class TestGenericStorage:
    """Test generic storage implementation using Claude credentials."""

    @pytest.mark.asyncio
    async def test_generic_storage_save_and_load_claude(self, tmp_path: object) -> None:
        """Test saving and loading Claude credentials."""
        storage_path = tmp_path / "test_claude.json"  # type: ignore[operator]
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        # Create test credentials
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_token"),
            refreshToken=SecretStr("refresh_token"),
            expiresAt=1234567890000,
            subscriptionType=None,
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)

        # Save
        assert await storage.save(credentials) is True
        assert storage_path.exists()

        # Load
        loaded = await storage.load()
        assert loaded is not None
        assert loaded.claude_ai_oauth.access_token.get_secret_value() == "test_token"
        assert (
            loaded.claude_ai_oauth.refresh_token.get_secret_value() == "refresh_token"
        )
        assert loaded.claude_ai_oauth.expires_at == 1234567890000

    @pytest.mark.asyncio
    async def test_generic_storage_load_nonexistent(self, tmp_path: object) -> None:
        """Test loading from nonexistent file returns None."""
        storage_path = tmp_path / "nonexistent.json"  # type: ignore[operator]
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        loaded = await storage.load()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_generic_storage_invalid_json(self, tmp_path: object) -> None:
        """Test loading invalid JSON returns None."""
        storage_path = tmp_path / "invalid.json"  # type: ignore[operator]
        storage_path.write_text("not valid json")
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        loaded = await storage.load()
        assert loaded is None


class TestTokenManagers:
    """Test refactored token managers."""

    @pytest.mark.asyncio
    async def test_claude_manager_with_generic_storage(self, tmp_path: object) -> None:
        """Test ClaudeApiTokenManager with GenericJsonStorage."""
        from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager

        storage_path = tmp_path / "claude_test.json"  # type: ignore[operator]
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)
        manager = ClaudeApiTokenManager(storage=storage)

        # Create and save credentials
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_token"),
            refreshToken=SecretStr("refresh_token"),
            expiresAt=int(datetime.now(UTC).timestamp() * 1000) + 3600000,
            subscriptionType=None,
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)

        assert await manager.save_credentials(credentials) is True

        # Load and verify
        loaded = await manager.load_credentials()
        assert loaded is not None
        assert manager.is_expired(loaded) is False
        assert await manager.get_access_token_value() == "test_token"

        snapshot = await manager.get_token_snapshot()
        assert snapshot is not None
        assert snapshot.provider == "claude-api"
        assert snapshot.access_token == "test_token"
        assert snapshot.refresh_token == "refresh_token"
        assert snapshot.expires_at is not None

    @pytest.mark.asyncio
    async def test_claude_manager_refreshes_before_expiry(
        self, tmp_path: object
    ) -> None:
        """Manager proactively refreshes when the token is nearing expiry."""
        from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager

        storage_path = tmp_path / "claude_refresh.json"  # type: ignore[operator]
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)
        manager = ClaudeApiTokenManager(storage=storage)

        near_expiry_ms = int(
            (datetime.now(UTC) + timedelta(seconds=45)).timestamp() * 1000
        )
        initial_credentials = ClaudeCredentials(
            claudeAiOauth=ClaudeOAuthToken(
                accessToken=SecretStr("stale_token"),
                refreshToken=SecretStr("refresh_token"),
                expiresAt=near_expiry_ms,
                subscriptionType=None,
            )
        )
        await manager.save_credentials(initial_credentials)

        refreshed_credentials = ClaudeCredentials(
            claudeAiOauth=ClaudeOAuthToken(
                accessToken=SecretStr("refreshed_token"),
                refreshToken=SecretStr("refresh_token"),
                expiresAt=int(
                    (datetime.now(UTC) + timedelta(hours=2)).timestamp() * 1000
                ),
                subscriptionType=None,
            )
        )

        async def _refresh() -> ClaudeCredentials:
            await manager.save_credentials(refreshed_credentials)
            return refreshed_credentials

        manager.refresh_token = AsyncMock(side_effect=_refresh)  # type: ignore[method-assign]

        token = await manager.get_access_token()

        assert token == "refreshed_token"
        assert manager.refresh_token.await_count == 1
        stored = await manager.load_credentials()
        assert stored is not None
        assert (
            stored.claude_ai_oauth.access_token.get_secret_value() == "refreshed_token"
        )

    @pytest.mark.asyncio
    async def test_claude_manager_raises_on_refresh_failure(
        self, tmp_path: object
    ) -> None:
        """Manager raises a consistent error when refresh fails."""
        from ccproxy.plugins.oauth_claude.manager import ClaudeApiTokenManager

        storage_path = tmp_path / "claude_refresh_fail.json"  # type: ignore[operator]
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)
        manager = ClaudeApiTokenManager(storage=storage)

        near_expiry_ms = int(
            (datetime.now(UTC) + timedelta(seconds=45)).timestamp() * 1000
        )
        credentials = ClaudeCredentials(
            claudeAiOauth=ClaudeOAuthToken(
                accessToken=SecretStr("stale_token"),
                refreshToken=SecretStr("refresh_token"),
                expiresAt=near_expiry_ms,
                subscriptionType=None,
            )
        )
        await manager.save_credentials(credentials)

        manager.refresh_token = AsyncMock(return_value=None)  # type: ignore[method-assign]

        with pytest.raises(OAuthTokenRefreshError):
            await manager.get_access_token_with_refresh()

        stored = await manager.load_credentials()
        assert stored is not None
        assert stored.claude_ai_oauth.access_token.get_secret_value() == "stale_token"


class TestUnifiedProfiles:
    """Test unified profile support in base manager."""

    @pytest.mark.asyncio
    async def test_get_unified_profile_with_new_format(self) -> None:
        """Test get_unified_profile with new BaseProfileInfo format."""

        # Create mock manager
        manager = MagicMock(spec=BaseTokenManager)

        # Create mock profile
        mock_profile = ClaudeProfileInfo(
            account_id="test-123",
            email="user@example.com",
            display_name="Test User",
            extras={"subscription": "pro"},
        )

        # Mock get_profile to return our profile
        async def mock_get_profile():
            return mock_profile

        manager.get_profile = mock_get_profile

        # Call get_unified_profile (bind the method to our mock)
        unified = await BaseTokenManager.get_unified_profile(manager)

        assert unified["account_id"] == "test-123"
        assert unified["email"] == "user@example.com"
        assert unified["display_name"] == "Test User"
        assert unified["provider"] == "claude-api"
        assert unified["extras"] == {"subscription": "pro"}
