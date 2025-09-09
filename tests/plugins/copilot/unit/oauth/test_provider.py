"""Unit tests for CopilotOAuthProvider."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from ccproxy.auth.oauth.protocol import StandardProfileFields
from ccproxy.plugins.copilot.config import CopilotOAuthConfig
from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotProfileInfo,
    CopilotTokenInfo,
    CopilotTokenResponse,
    DeviceCodeResponse,
)
from ccproxy.plugins.copilot.oauth.provider import CopilotOAuthProvider
from ccproxy.plugins.copilot.oauth.storage import CopilotOAuthStorage


class TestCopilotOAuthProvider:
    """Test cases for CopilotOAuthProvider."""

    @pytest.fixture
    def mock_config(self) -> CopilotOAuthConfig:
        """Create mock OAuth configuration."""
        return CopilotOAuthConfig(
            client_id="test-client-id",
            authorize_url="https://github.com/login/device/code",
            token_url="https://github.com/login/oauth/access_token",
            copilot_token_url="https://api.github.com/copilot_internal/v2/token",
            scopes=["read:user"],
            use_pkce=True,
            account_type="individual",
        )

    @pytest.fixture
    def mock_storage(self) -> CopilotOAuthStorage:
        """Create mock storage."""
        storage = MagicMock(spec=CopilotOAuthStorage)
        storage.load = AsyncMock(return_value=None)
        storage.save = AsyncMock()
        storage.delete = AsyncMock()
        storage.load_credentials = AsyncMock(return_value=None)
        storage.clear_credentials = AsyncMock()
        return storage

    @pytest.fixture
    def mock_http_client(self) -> MagicMock:
        """Create mock HTTP client."""
        return MagicMock()

    @pytest.fixture
    def mock_hook_manager(self) -> MagicMock:
        """Create mock hook manager."""
        return MagicMock()

    @pytest.fixture
    def mock_detection_service(self) -> MagicMock:
        """Create mock CLI detection service."""
        return MagicMock()

    @pytest.fixture
    def oauth_provider(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
        mock_http_client: MagicMock,
        mock_hook_manager: MagicMock,
        mock_detection_service: MagicMock,
    ) -> CopilotOAuthProvider:
        """Create CopilotOAuthProvider instance."""
        return CopilotOAuthProvider(
            config=mock_config,
            storage=mock_storage,
            http_client=mock_http_client,
            hook_manager=mock_hook_manager,
            detection_service=mock_detection_service,
        )

    @pytest.fixture
    def mock_oauth_token(self) -> CopilotOAuthToken:
        """Create mock OAuth token."""
        now = int(datetime.now(UTC).timestamp())
        return CopilotOAuthToken(
            access_token=SecretStr("gho_test_token"),
            token_type="bearer",
            expires_in=28800,  # 8 hours
            created_at=now,
            scope="read:user",
        )

    @pytest.fixture
    def mock_copilot_token(self) -> CopilotTokenResponse:
        """Create mock Copilot token."""
        expires_at = (datetime.now(UTC) + timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        return CopilotTokenResponse(
            token=SecretStr("copilot_test_token"),
            expires_at=expires_at,
        )

    @pytest.fixture
    def mock_credentials(
        self,
        mock_oauth_token: CopilotOAuthToken,
        mock_copilot_token: CopilotTokenResponse,
    ) -> CopilotCredentials:
        """Create mock credentials."""
        return CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=mock_copilot_token,
            account_type="individual",
        )

    def test_init_with_defaults(self) -> None:
        """Test initialization with default values."""
        provider = CopilotOAuthProvider()

        assert isinstance(provider.config, CopilotOAuthConfig)
        assert isinstance(provider.storage, CopilotOAuthStorage)
        assert provider.hook_manager is None
        assert provider.detection_service is None
        assert provider.http_client is None
        assert provider._cached_profile is None

    def test_init_with_custom_values(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
        mock_http_client: MagicMock,
        mock_hook_manager: MagicMock,
        mock_detection_service: MagicMock,
    ) -> None:
        """Test initialization with custom values."""
        provider = CopilotOAuthProvider(
            config=mock_config,
            storage=mock_storage,
            http_client=mock_http_client,
            hook_manager=mock_hook_manager,
            detection_service=mock_detection_service,
        )

        assert provider.config is mock_config
        assert provider.storage is mock_storage
        assert provider.http_client is mock_http_client
        assert provider.hook_manager is mock_hook_manager
        assert provider.detection_service is mock_detection_service

    def test_provider_properties(self, oauth_provider: CopilotOAuthProvider) -> None:
        """Test provider properties."""
        assert oauth_provider.provider_name == "copilot"
        assert oauth_provider.provider_display_name == "GitHub Copilot"
        assert oauth_provider.supports_pkce is True
        assert oauth_provider.supports_refresh is True
        assert oauth_provider.requires_client_secret is False

    async def test_get_authorization_url(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test getting authorization URL."""
        url = await oauth_provider.get_authorization_url("test-state", "test-verifier")

        assert url == "https://github.com/login/device/code"

    async def test_start_device_flow(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test starting device flow."""
        mock_response = DeviceCodeResponse(
            device_code="test-device-code",
            user_code="ABCD-1234",
            verification_uri="https://github.com/login/device",
            expires_in=900,
            interval=5,
        )

        with patch.object(
            oauth_provider.client, "start_device_flow", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = mock_response

            (
                device_code,
                user_code,
                verification_uri,
                expires_in,
            ) = await oauth_provider.start_device_flow()

            assert device_code == "test-device-code"
            assert user_code == "ABCD-1234"
            assert verification_uri == "https://github.com/login/device"
            assert expires_in == 900

    async def test_complete_device_flow(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test completing device flow."""
        mock_credentials = MagicMock(spec=CopilotCredentials)

        with patch.object(
            oauth_provider.client, "complete_authorization", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = mock_credentials

            result = await oauth_provider.complete_device_flow(
                "test-device-code", 5, 900
            )

            assert result is mock_credentials
            mock_client.assert_called_once_with("test-device-code", 5, 900)

    async def test_exchange_code_not_implemented(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test that exchange_code raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError,
            match="Device code flow doesn't use authorization code exchange",
        ):
            await oauth_provider.exchange_code("test-code", "test-state")

    async def test_refresh_token_success(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful token refresh."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        refreshed_credentials = MagicMock(spec=CopilotCredentials)
        refreshed_credentials.copilot_token = mock_credentials.copilot_token

        with patch.object(
            oauth_provider.client, "refresh_copilot_token", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = refreshed_credentials

            result = await oauth_provider.refresh_token("dummy-refresh-token")

            assert result["access_token"] == "copilot_test_token"
            assert result["token_type"] == "bearer"
            assert result["provider"] == "copilot"
            assert "expires_at" in result

    async def test_refresh_token_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test token refresh when no credentials found."""
        oauth_provider.storage.load_credentials.return_value = None

        with pytest.raises(ValueError, match="No credentials found for refresh"):
            await oauth_provider.refresh_token("dummy-refresh-token")

    async def test_refresh_token_no_copilot_token(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test token refresh when Copilot token is None."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        refreshed_credentials = MagicMock(spec=CopilotCredentials)
        refreshed_credentials.copilot_token = None

        with patch.object(
            oauth_provider.client, "refresh_copilot_token", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = refreshed_credentials

            with pytest.raises(ValueError, match="Failed to refresh Copilot token"):
                await oauth_provider.refresh_token("dummy-refresh-token")

    async def test_get_user_profile_success(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful user profile retrieval."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        mock_profile = CopilotProfileInfo(
            account_id="12345",
            provider_type="copilot",
            login="testuser",
            name="Test User",
            email="test@example.com",
        )

        with patch.object(
            oauth_provider.client, "get_user_profile", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = mock_profile

            result = await oauth_provider.get_user_profile("test-token")

            assert isinstance(result, StandardProfileFields)
            assert result.account_id == "12345"
            assert result.provider_type == "copilot"
            assert result.email == "test@example.com"
            assert result.display_name == "Test User"

    async def test_get_user_profile_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test user profile retrieval when no credentials found."""
        oauth_provider.storage.load_credentials.return_value = None

        with pytest.raises(ValueError, match="No credentials found"):
            await oauth_provider.get_user_profile("test-token")

    async def test_get_token_info_success(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful token info retrieval."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        # Mock get_user_profile to return a profile
        mock_profile = StandardProfileFields(
            account_id="12345",
            provider_type="copilot",
            email="test@example.com",
            display_name="Test User",
        )

        with patch.object(
            oauth_provider, "get_user_profile", new_callable=AsyncMock
        ) as mock_get_profile:
            mock_get_profile.return_value = mock_profile

            result = await oauth_provider.get_token_info()

            assert isinstance(result, CopilotTokenInfo)
            assert result.provider == "copilot"
            assert result.account_type == "individual"
            assert result.oauth_expires_at is not None
            assert result.copilot_expires_at is not None

    async def test_get_token_info_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test token info retrieval when no credentials found."""
        oauth_provider.storage.load_credentials.return_value = None

        result = await oauth_provider.get_token_info()

        assert result is None

    async def test_is_authenticated_with_valid_tokens(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test authentication check with valid tokens."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.is_authenticated()

        assert result is True

    async def test_is_authenticated_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test authentication check when no credentials found."""
        oauth_provider.storage.load_credentials.return_value = None

        result = await oauth_provider.is_authenticated()

        assert result is False

    async def test_is_authenticated_expired_oauth_token(
        self,
        oauth_provider: CopilotOAuthProvider,
    ) -> None:
        """Test authentication check with expired OAuth token."""
        # Create expired OAuth token
        past_time = int((datetime.now(UTC) - timedelta(days=1)).timestamp())
        expired_oauth_token = CopilotOAuthToken(
            access_token=SecretStr("gho_test_token"),
            token_type="bearer",
            expires_in=3600,  # 1 hour
            created_at=past_time - 3600,  # Created and expired yesterday
            scope="read:user",
        )

        mock_credentials = CopilotCredentials(
            oauth_token=expired_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.is_authenticated()

        assert result is False

    async def test_is_authenticated_no_copilot_token(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test authentication check when no Copilot token."""
        mock_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.is_authenticated()

        assert result is False

    async def test_get_copilot_token_success(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful Copilot token retrieval."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.get_copilot_token()

        assert result == "copilot_test_token"

    async def test_get_copilot_token_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test Copilot token retrieval when no credentials."""
        oauth_provider.storage.load_credentials.return_value = None

        result = await oauth_provider.get_copilot_token()

        assert result is None

    async def test_get_copilot_token_no_copilot_token(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test Copilot token retrieval when no Copilot token."""
        mock_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.get_copilot_token()

        assert result is None

    async def test_ensure_copilot_token_success(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful Copilot token ensure."""
        oauth_provider.storage.load_credentials.return_value = mock_credentials

        result = await oauth_provider.ensure_copilot_token()

        assert result == "copilot_test_token"

    async def test_ensure_copilot_token_no_credentials(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test ensure Copilot token when no credentials."""
        oauth_provider.storage.load_credentials.return_value = None

        with pytest.raises(
            ValueError, match="No credentials found - authorization required"
        ):
            await oauth_provider.ensure_copilot_token()

    async def test_ensure_copilot_token_expired_oauth(
        self,
        oauth_provider: CopilotOAuthProvider,
    ) -> None:
        """Test ensure Copilot token with expired OAuth token."""
        # Create expired OAuth token
        past_time = int((datetime.now(UTC) - timedelta(days=1)).timestamp())
        expired_oauth_token = CopilotOAuthToken(
            access_token=SecretStr("gho_test_token"),
            token_type="bearer",
            expires_in=3600,  # 1 hour
            created_at=past_time - 3600,  # Created and expired yesterday
            scope="read:user",
        )

        mock_credentials = CopilotCredentials(
            oauth_token=expired_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = mock_credentials

        with pytest.raises(
            ValueError, match="OAuth token expired - re-authorization required"
        ):
            await oauth_provider.ensure_copilot_token()

    async def test_ensure_copilot_token_refresh_needed(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test ensure Copilot token when refresh is needed."""
        mock_credentials_no_copilot = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        mock_copilot_token = CopilotTokenResponse(
            token=SecretStr("refreshed_copilot_token"),
            expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat() + "Z",
        )

        mock_refreshed_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=mock_copilot_token,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = (
            mock_credentials_no_copilot
        )

        with patch.object(
            oauth_provider.client, "refresh_copilot_token", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = mock_refreshed_credentials

            result = await oauth_provider.ensure_copilot_token()

            assert result == "refreshed_copilot_token"

    async def test_ensure_copilot_token_refresh_failed(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test ensure Copilot token when refresh fails."""
        mock_credentials_no_copilot = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        mock_failed_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,  # Still no copilot token after refresh
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = (
            mock_credentials_no_copilot
        )

        with patch.object(
            oauth_provider.client, "refresh_copilot_token", new_callable=AsyncMock
        ) as mock_client:
            mock_client.return_value = mock_failed_credentials

            with pytest.raises(ValueError, match="Failed to obtain Copilot token"):
                await oauth_provider.ensure_copilot_token()

    async def test_logout(self, oauth_provider: CopilotOAuthProvider) -> None:
        """Test logout functionality."""
        await oauth_provider.logout()

        oauth_provider.storage.clear_credentials.assert_called_once()

    async def test_cleanup_success(self, oauth_provider: CopilotOAuthProvider) -> None:
        """Test successful cleanup."""
        oauth_provider.client.close = AsyncMock()

        await oauth_provider.cleanup()

        oauth_provider.client.close.assert_called_once()

    async def test_cleanup_with_error(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test cleanup with error."""
        oauth_provider.client.close = AsyncMock(side_effect=Exception("Test error"))

        # Should not raise exception, just log the error
        await oauth_provider.cleanup()

        oauth_provider.client.close.assert_called_once()

    def test_get_provider_info(self, oauth_provider: CopilotOAuthProvider) -> None:
        """Test getting provider info."""
        info = oauth_provider.get_provider_info()

        assert info.name == "copilot"
        assert info.display_name == "GitHub Copilot"
        assert info.description == "GitHub Copilot OAuth authentication"
        assert info.supports_pkce is True
        assert info.scopes == ["read:user", "copilot"]
        assert info.is_available is True
        assert info.plugin_name == "copilot"

    def test_extract_standard_profile_from_profile_info(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test extracting standard profile from CopilotProfileInfo."""
        profile_info = CopilotProfileInfo(
            account_id="12345",
            provider_type="copilot",
            login="testuser",
            name="Test User",
            email="test@example.com",
        )

        result = oauth_provider._extract_standard_profile(profile_info)

        assert isinstance(result, StandardProfileFields)
        assert result.account_id == "12345"
        assert result.provider_type == "copilot"
        assert result.email == "test@example.com"
        assert result.display_name == "Test User"

    def test_extract_standard_profile_from_credentials(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test extracting standard profile from CopilotCredentials."""
        result = oauth_provider._extract_standard_profile(mock_credentials)

        assert isinstance(result, StandardProfileFields)
        assert result.account_id == "unknown"
        assert result.provider_type == "copilot"
        assert result.email is None
        assert result.display_name == "GitHub Copilot User"

    def test_extract_standard_profile_from_unknown(
        self, oauth_provider: CopilotOAuthProvider
    ) -> None:
        """Test extracting standard profile from unknown object."""
        result = oauth_provider._extract_standard_profile("unknown")

        assert isinstance(result, StandardProfileFields)
        assert result.account_id == "unknown"
        assert result.provider_type == "copilot"
        assert result.email is None
        assert result.display_name == "Unknown User"

    async def test_copilot_token_expiration_check(
        self,
        oauth_provider: CopilotOAuthProvider,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test that expired Copilot tokens are detected and refreshed."""
        from datetime import UTC, datetime

        from ccproxy.plugins.copilot.oauth.models import CopilotTokenResponse

        # Create an expired Copilot token (1 hour ago)
        expired_time = datetime.now(UTC).timestamp() - 3600
        expired_copilot_token = CopilotTokenResponse(
            token="expired_copilot_token",
            expires_at=int(expired_time),
            refresh_in=3600,
        )

        # Create credentials with expired Copilot token
        mock_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=expired_copilot_token,
            account_type="individual",
        )

        oauth_provider.storage.load_credentials.return_value = mock_credentials

        # Mock the refresh to return new token
        new_copilot_token = CopilotTokenResponse(
            token="new_copilot_token",
            expires_at=int(datetime.now(UTC).timestamp() + 3600),  # 1 hour from now
            refresh_in=3600,
        )
        new_credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=new_copilot_token,
            account_type="individual",
        )

        # Verify the expired token is detected as expired
        assert expired_copilot_token.is_expired is True

        # Verify get_copilot_token returns None for expired token
        token = await oauth_provider.get_copilot_token()
        assert token is None

        # Verify is_authenticated returns False for expired token
        is_auth = await oauth_provider.is_authenticated()
        assert is_auth is False

        # Verify ensure_copilot_token refreshes expired token
        with patch.object(
            oauth_provider.client, "refresh_copilot_token", new_callable=AsyncMock
        ) as mock_refresh:
            mock_refresh.return_value = new_credentials

            result = await oauth_provider.ensure_copilot_token()
            assert result == "new_copilot_token"
            mock_refresh.assert_called_once()
