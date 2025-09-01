"""Tests for refactored authentication components."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo
from ccproxy.auth.storage.generic import GenericJsonStorage
from plugins.oauth_claude.models import (
    ClaudeCredentials,
    ClaudeOAuthToken,
    ClaudeProfileInfo,
    ClaudeTokenWrapper,
)


class TestBaseModels:
    """Test base authentication models."""

    def test_base_token_info_is_expired(self):
        """Test that is_expired computed field works correctly."""

        class TestToken(BaseTokenInfo):
            test_expires_at: datetime

            @property
            def access_token_value(self) -> str:
                return "test_token"

            @property
            def expires_at_datetime(self) -> datetime:
                return self.test_expires_at

        # Test expired token
        expired_token = TestToken(
            test_expires_at=datetime.now(UTC) - timedelta(hours=1)
        )
        assert expired_token.is_expired is True

        # Test valid token
        valid_token = TestToken(test_expires_at=datetime.now(UTC) + timedelta(hours=1))
        assert valid_token.is_expired is False

    def test_base_profile_info(self):
        """Test BaseProfileInfo model."""
        profile = BaseProfileInfo(
            account_id="test_id",
            provider_type="test_provider",
            email="test@example.com",
            display_name="Test User",
            extras={"custom": "data"},
        )

        assert profile.account_id == "test_id"
        assert profile.provider_type == "test_provider"
        assert profile.email == "test@example.com"
        assert profile.display_name == "Test User"
        assert profile.extras == {"custom": "data"}


class TestClaudeModels:
    """Test Claude-specific models."""

    def test_claude_token_wrapper(self):
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

        # Create wrapper
        wrapper = ClaudeTokenWrapper(credentials=credentials)

        # Test properties
        assert wrapper.access_token_value == "test_access"
        assert wrapper.refresh_token_value == "test_refresh"
        assert wrapper.is_expired is False
        assert wrapper.subscription_type == "pro"
        assert wrapper.scopes == ["read", "write"]

    def test_claude_token_wrapper_expired(self):
        """Test ClaudeTokenWrapper with expired token."""
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_access"),
            refreshToken=SecretStr("test_refresh"),
            expiresAt=int(datetime.now(UTC).timestamp() * 1000) - 3600000,  # 1 hour ago
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)
        wrapper = ClaudeTokenWrapper(credentials=credentials)

        assert wrapper.is_expired is True

    def test_claude_profile_from_api_response(self):
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


class TestOpenAIModels:
    """Deprecated tests removed; OpenAI model has a new schema."""

    pass


class TestGenericStorage:
    """Test generic storage implementation."""

    @pytest.mark.asyncio
    async def test_generic_storage_save_and_load_claude(self, tmp_path):
        """Test saving and loading Claude credentials."""
        storage_path = tmp_path / "test_claude.json"
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        # Create test credentials
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_token"),
            refreshToken=SecretStr("refresh_token"),
            expiresAt=1234567890000,
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
    async def test_generic_storage_save_and_load_openai(self, tmp_path):
        """Deprecated legacy OpenAI storage test removed."""
        pass

    @pytest.mark.asyncio
    async def test_generic_storage_load_nonexistent(self, tmp_path):
        """Test loading from nonexistent file returns None."""
        storage_path = tmp_path / "nonexistent.json"
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        loaded = await storage.load()
        assert loaded is None

    @pytest.mark.asyncio
    async def test_generic_storage_invalid_json(self, tmp_path):
        """Test loading invalid JSON returns None."""
        storage_path = tmp_path / "invalid.json"
        storage_path.write_text("not valid json")
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        loaded = await storage.load()
        assert loaded is None


class TestTokenManagers:
    """Test refactored token managers."""

    @pytest.mark.asyncio
    async def test_claude_manager_with_generic_storage(self, tmp_path):
        """Test ClaudeApiTokenManager with GenericJsonStorage."""
        from plugins.oauth_claude.manager import ClaudeApiTokenManager

        storage_path = tmp_path / "claude_test.json"
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)
        manager = ClaudeApiTokenManager(storage=storage)

        # Create and save credentials
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_token"),
            refreshToken=SecretStr("refresh_token"),
            expiresAt=int(datetime.now(UTC).timestamp() * 1000) + 3600000,
        )
        credentials = ClaudeCredentials(claudeAiOauth=oauth)

        assert await manager.save_credentials(credentials) is True

        # Load and verify
        loaded = await manager.load_credentials()
        assert loaded is not None
        assert manager.is_expired(loaded) is False
        assert await manager.get_access_token_value() == "test_token"

    @pytest.mark.asyncio
    async def test_codex_manager_with_generic_storage(self, tmp_path):
        """Deprecated legacy Codex manager test removed."""
        pass

    @pytest.mark.asyncio
    @patch("jwt.decode")
    async def test_codex_manager_profile_extraction(self, mock_decode, tmp_path):
        """Deprecated legacy profile extraction test removed."""
        pass


class TestUnifiedProfiles:
    """Test unified profile support in base manager."""

    @pytest.mark.asyncio
    async def test_get_unified_profile_with_new_format(self):
        """Test get_unified_profile with new BaseProfileInfo format."""
        from ccproxy.auth.managers.base import BaseTokenManager

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
