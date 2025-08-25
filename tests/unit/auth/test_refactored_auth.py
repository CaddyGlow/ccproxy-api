"""Tests for refactored authentication components."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ccproxy.auth.models import ClaudeCredentials, OAuthToken, OpenAICredentials
from ccproxy.auth.models.base import BaseProfileInfo, BaseTokenInfo
from ccproxy.auth.storage.generic import GenericJsonStorage
from plugins.claude_api.auth.models import ClaudeProfileInfo, ClaudeTokenWrapper
from plugins.codex.auth.models import OpenAIProfileInfo, OpenAITokenWrapper


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
        oauth = OAuthToken(
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
        oauth = OAuthToken(
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
    """Test OpenAI/Codex-specific models."""

    def test_openai_token_wrapper(self):
        """Test OpenAITokenWrapper functionality."""
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        credentials = OpenAICredentials(
            access_token="test_access",
            refresh_token="test_refresh",
            id_token="test_id_token",
            expires_at=expires_at,
            account_id="test_account",
        )

        wrapper = OpenAITokenWrapper(credentials=credentials)

        assert wrapper.access_token_value == "test_access"
        assert wrapper.refresh_token_value == "test_refresh"
        assert wrapper.expires_at_datetime == expires_at
        assert wrapper.account_id == "test_account"
        assert wrapper.id_token == "test_id_token"
        assert wrapper.is_expired is False

    @patch("jwt.decode")
    def test_openai_profile_from_token(self, mock_decode):
        """Test creating OpenAIProfileInfo from JWT token."""
        # Mock JWT claims
        mock_claims = {
            "email": "user@openai.com",
            "name": "OpenAI User",
            "sub": "auth0|123456",
            "org_id": "org-123",
            "https://api.openai.com/auth": {
                "chatgpt_account_id": "chatgpt-uuid",
                "organization_id": "org-456",
            },
        }
        mock_decode.return_value = mock_claims

        credentials = OpenAICredentials(
            access_token="mock_jwt",
            refresh_token="refresh",
            expires_at=datetime.now(UTC),
            account_id="test_account",
        )

        profile = OpenAIProfileInfo.from_token(credentials)

        assert profile.account_id == "test_account"
        assert profile.email == "user@openai.com"
        assert profile.display_name == "OpenAI User"
        assert profile.provider_type == "openai"
        assert profile.chatgpt_account_id == "chatgpt-uuid"
        assert profile.organization_id == "org-456"
        assert profile.auth0_subject == "auth0|123456"
        assert profile.extras == mock_claims


class TestGenericStorage:
    """Test generic storage implementation."""

    @pytest.mark.asyncio
    async def test_generic_storage_save_and_load_claude(self, tmp_path):
        """Test saving and loading Claude credentials."""
        storage_path = tmp_path / "test_claude.json"
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)

        # Create test credentials
        oauth = OAuthToken(
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
        """Test saving and loading OpenAI credentials."""
        storage_path = tmp_path / "test_openai.json"
        storage = GenericJsonStorage(storage_path, OpenAICredentials)

        # Create test credentials
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        credentials = OpenAICredentials(
            access_token="access_token",
            refresh_token="refresh_token",
            id_token="id_token",
            expires_at=expires_at,
            account_id="account_123",
        )

        # Save
        assert await storage.save(credentials) is True
        assert storage_path.exists()

        # Load
        loaded = await storage.load()
        assert loaded is not None
        assert loaded.access_token == "access_token"
        assert loaded.refresh_token == "refresh_token"
        assert loaded.id_token == "id_token"
        assert loaded.account_id == "account_123"
        # Check expiration is close (within 1 second)
        assert abs((loaded.expires_at - expires_at).total_seconds()) < 1

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
        from plugins.claude_api.auth.manager import ClaudeApiTokenManager

        storage_path = tmp_path / "claude_test.json"
        storage = GenericJsonStorage(storage_path, ClaudeCredentials)
        manager = ClaudeApiTokenManager(storage=storage)

        # Create and save credentials
        oauth = OAuthToken(
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
        """Test CodexTokenManager with GenericJsonStorage."""
        from plugins.codex.auth.manager import CodexTokenManager

        storage_path = tmp_path / "openai_test.json"
        storage = GenericJsonStorage(storage_path, OpenAICredentials)
        manager = CodexTokenManager(storage=storage)

        # Create and save credentials
        expires_at = datetime.now(UTC) + timedelta(hours=1)
        credentials = OpenAICredentials(
            access_token="test_token",
            refresh_token="refresh_token",
            expires_at=expires_at,
            account_id="test_account",
        )

        assert await manager.save_credentials(credentials) is True

        # Load and verify
        loaded = await manager.load_credentials()
        assert loaded is not None
        assert manager.is_expired(loaded) is False
        assert manager.get_account_id(loaded) == "test_account"

    @pytest.mark.asyncio
    @patch("jwt.decode")
    async def test_codex_manager_profile_extraction(self, mock_decode, tmp_path):
        """Test CodexTokenManager profile extraction from JWT."""
        from plugins.codex.auth.manager import CodexTokenManager

        # Mock JWT claims
        mock_claims = {
            "email": "test@openai.com",
            "name": "Test User",
            "https://api.openai.com/auth": {"chatgpt_account_id": "chatgpt-123"},
        }
        mock_decode.return_value = mock_claims

        storage_path = tmp_path / "openai_test.json"
        storage = GenericJsonStorage(storage_path, OpenAICredentials)
        manager = CodexTokenManager(storage=storage)

        # Save credentials
        credentials = OpenAICredentials(
            access_token="mock_jwt",
            refresh_token="refresh",
            expires_at=datetime.now(UTC) + timedelta(hours=1),
            account_id="test_account",
        )
        await manager.save_credentials(credentials)

        # Get profile
        profile = await manager.get_profile()
        assert profile is not None
        assert profile.account_id == "test_account"
        assert profile.email == "test@openai.com"
        assert profile.display_name == "Test User"


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
