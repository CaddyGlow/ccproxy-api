"""Unit tests for CopilotOAuthClient."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import SecretStr

from ccproxy.plugins.copilot.config import CopilotOAuthConfig
from ccproxy.plugins.copilot.oauth.client import CopilotOAuthClient
from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotProfileInfo,
    CopilotTokenResponse,
    DeviceCodeResponse,
)
from ccproxy.plugins.copilot.oauth.storage import CopilotOAuthStorage


class TestCopilotOAuthClient:
    """Test cases for CopilotOAuthClient."""

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
        )

    @pytest.fixture
    def mock_storage(self) -> CopilotOAuthStorage:
        """Create mock storage."""
        storage = MagicMock(spec=CopilotOAuthStorage)
        storage.store_credentials = AsyncMock()
        storage.load_credentials = AsyncMock(return_value=None)
        return storage

    @pytest.fixture
    def mock_http_client(self) -> MagicMock:
        """Create mock HTTP client."""
        return MagicMock()

    def test_init_with_defaults(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test client initialization with default parameters."""
        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        assert client.config is mock_config
        assert client.storage is mock_storage
        assert client.hook_manager is None
        assert client.detection_service is None
        assert client._http_client is None
        assert client._owns_client is True

    def test_init_with_all_parameters(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
        mock_http_client: MagicMock,
    ) -> None:
        """Test client initialization with all parameters."""
        mock_hook_manager = MagicMock()
        mock_detection_service = MagicMock()

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
            http_client=mock_http_client,
            hook_manager=mock_hook_manager,
            detection_service=mock_detection_service,
        )

        assert client.config is mock_config
        assert client.storage is mock_storage
        assert client.hook_manager is mock_hook_manager
        assert client.detection_service is mock_detection_service
        assert client._http_client is mock_http_client
        assert client._owns_client is False

    async def test_get_http_client_creates_default(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test HTTP client creation when none provided."""
        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        http_client = await client._get_http_client()

        assert http_client is not None
        assert isinstance(http_client, httpx.AsyncClient)
        assert client._http_client is http_client

        # Clean up
        await client.close()

    async def test_get_http_client_returns_existing(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
        mock_http_client: MagicMock,
    ) -> None:
        """Test HTTP client returns existing when provided."""
        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
            http_client=mock_http_client,
        )

        http_client = await client._get_http_client()

        assert http_client is mock_http_client

    async def test_start_device_flow_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful device flow start."""
        mock_response_data = {
            "device_code": "test-device-code",
            "user_code": "ABCD-1234",
            "verification_uri": "https://github.com/login/device",
            "verification_uri_complete": "https://github.com/login/device?user_code=ABCD-1234",
            "expires_in": 900,
            "interval": 5,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with patch.object(client, "_get_http_client", return_value=mock_client):
            result = await client.start_device_flow()

        assert isinstance(result, DeviceCodeResponse)
        assert result.device_code == "test-device-code"
        assert result.user_code == "ABCD-1234"
        assert result.verification_uri == "https://github.com/login/device"
        assert result.expires_in == 900

        mock_client.post.assert_called_once_with(
            mock_config.authorize_url,
            data={
                "client_id": mock_config.client_id,
                "scope": " ".join(mock_config.scopes),
            },
            headers={"Accept": "application/json"},
        )

    async def test_start_device_flow_http_error(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test device flow start with HTTP error."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = httpx.HTTPError("Network error")

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPError),
        ):
            await client.start_device_flow()

    async def test_poll_for_token_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful token polling."""
        mock_response_data = {
            "access_token": "test-access-token",
            "token_type": "bearer",
            "scope": "read:user",
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with patch.object(client, "_get_http_client", return_value=mock_client):
            result = await client.poll_for_token("device-code", 1, 60)

        assert isinstance(result, CopilotOAuthToken)
        assert result.access_token.get_secret_value() == "test-access-token"
        assert result.token_type == "bearer"
        assert result.scope == "read:user"

    async def test_poll_for_token_pending(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test token polling with pending status."""
        # First response: pending
        pending_response = MagicMock()
        pending_response.json.return_value = {
            "error": "authorization_pending",
            "error_description": "The authorization request is still pending",
        }

        # Second response: success
        success_response = MagicMock()
        success_response.json.return_value = {
            "access_token": "test-token",
            "token_type": "bearer",
            "scope": "read:user",
        }

        mock_client = AsyncMock()
        mock_client.post.side_effect = [pending_response, success_response]

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            patch("asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await client.poll_for_token(
                "device-code", 0.01, 60
            )  # Much faster interval for tests

        assert isinstance(result, CopilotOAuthToken)
        assert result.access_token.get_secret_value() == "test-token"

    async def test_poll_for_token_expired(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test token polling with expired code."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "expired_token",
            "error_description": "The device code has expired",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            pytest.raises(TimeoutError, match="Device code has expired"),
        ):
            await client.poll_for_token(
                "device-code", 0.01, 60
            )  # Much faster interval for tests

    async def test_poll_for_token_denied(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test token polling with access denied."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "error": "access_denied",
            "error_description": "The user has denied the request",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            pytest.raises(ValueError, match="User denied authorization"),
        ):
            await client.poll_for_token(
                "device-code", 0.01, 60
            )  # Much faster interval for tests

    async def test_exchange_for_copilot_token_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful Copilot token exchange."""
        oauth_token = CopilotOAuthToken(
            access_token=SecretStr("github-token"),
            token_type="bearer",
            scope="read:user",
            created_at=int(datetime.now(UTC).timestamp()),
            expires_in=None,
        )

        mock_response_data = {
            "token": "copilot-service-token",
            "expires_at": "2024-12-31T23:59:59Z",
            "refresh_in": 3600,
        }

        mock_response = MagicMock()
        mock_response.json.return_value = mock_response_data
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with patch.object(client, "_get_http_client", return_value=mock_client):
            result = await client.exchange_for_copilot_token(oauth_token)

        assert isinstance(result, CopilotTokenResponse)
        assert result.token.get_secret_value() == "copilot-service-token"
        # expires_at is now converted to datetime object
        expected_dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
        assert result.expires_at == expected_dt

        mock_client.get.assert_called_once_with(
            mock_config.copilot_token_url,
            headers={
                "Authorization": "Bearer github-token",
                "Accept": "application/json",
            },
        )

    async def test_exchange_for_copilot_token_http_error(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test Copilot token exchange with HTTP error."""
        oauth_token = CopilotOAuthToken(
            access_token=SecretStr("github-token"),
            token_type="bearer",
            scope="read:user",
            created_at=int(datetime.now(UTC).timestamp()),
            expires_in=None,
        )

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.HTTPError("Service unavailable")

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            pytest.raises(httpx.HTTPError),
        ):
            await client.exchange_for_copilot_token(oauth_token)

    async def test_get_user_profile_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful user profile retrieval."""
        oauth_token = CopilotOAuthToken(
            access_token=SecretStr("github-token"),
            token_type="bearer",
            scope="read:user",
            created_at=int(datetime.now(UTC).timestamp()),
            expires_in=None,
        )

        # Mock user profile response
        user_response = MagicMock()
        user_response.json.return_value = {
            "id": 12345,
            "login": "testuser",
            "name": "Test User",
            "email": "test@example.com",
            "avatar_url": "https://avatar.example.com/testuser",
            "html_url": "https://github.com/testuser",
        }
        user_response.raise_for_status = MagicMock()

        # Mock Copilot individual response
        copilot_response = MagicMock()
        copilot_response.status_code = 200
        copilot_response.json.return_value = {"seat_breakdown": {"total": 1}}

        mock_client = AsyncMock()
        mock_client.get.side_effect = [
            user_response,
            MagicMock(status_code=404),  # Business accounts not found
            copilot_response,  # Individual plan found
        ]

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with (
            patch.object(client, "_get_http_client", return_value=mock_client),
            patch("ccproxy.core.logging.get_plugin_logger"),
        ):
            result = await client.get_user_profile(oauth_token)

        assert isinstance(result, CopilotProfileInfo)
        assert result.account_id == "12345"
        assert result.login == "testuser"
        assert result.name == "Test User"
        assert result.email == "test@example.com"
        assert result.copilot_access is True
        assert result.copilot_plan == "individual"

    async def test_complete_authorization_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful complete authorization flow."""
        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        # Mock the individual methods
        mock_oauth_token = CopilotOAuthToken(
            access_token=SecretStr("github-token"),
            token_type="bearer",
            scope="read:user",
            created_at=int(datetime.now(UTC).timestamp()),
            expires_in=None,
        )

        mock_copilot_token = CopilotTokenResponse(
            token=SecretStr("copilot-token"),
            expires_at="2024-12-31T23:59:59Z",
        )

        mock_profile = CopilotProfileInfo(
            account_id="12345",
            login="testuser",
            name="Test User",
            email="test@example.com",
            avatar_url="https://avatar.example.com/testuser",
            html_url="https://github.com/testuser",
            copilot_plan="individual",
            copilot_access=True,
        )

        with (
            patch.object(client, "poll_for_token", return_value=mock_oauth_token),
            patch.object(
                client, "exchange_for_copilot_token", return_value=mock_copilot_token
            ),
            patch.object(client, "get_user_profile", return_value=mock_profile),
        ):
            result = await client.complete_authorization("device-code", 5, 900)

        assert isinstance(result, CopilotCredentials)
        assert result.oauth_token is mock_oauth_token
        assert result.copilot_token is mock_copilot_token
        assert result.account_type == "individual"

        # Verify storage was called
        mock_storage.store_credentials.assert_called_once_with(result)

    async def test_refresh_copilot_token_success(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test successful Copilot token refresh."""
        oauth_token = CopilotOAuthToken(
            access_token=SecretStr("github-token"),
            token_type="bearer",
            scope="read:user",
            created_at=int(datetime.now(UTC).timestamp()),
            expires_in=None,
        )

        old_copilot_token = CopilotTokenResponse(
            token=SecretStr("old-copilot-token"),
            expires_at="2024-06-01T12:00:00Z",
        )

        credentials = CopilotCredentials(
            oauth_token=oauth_token,
            copilot_token=old_copilot_token,
            account_type="individual",
        )

        new_copilot_token = CopilotTokenResponse(
            token=SecretStr("new-copilot-token"),
            expires_at="2024-12-31T23:59:59Z",
        )

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        with patch.object(
            client, "exchange_for_copilot_token", return_value=new_copilot_token
        ):
            result = await client.refresh_copilot_token(credentials)

        assert result.copilot_token is new_copilot_token
        assert result.oauth_token is oauth_token  # Should remain same
        mock_storage.store_credentials.assert_called_once_with(result)

    async def test_close_with_owned_client(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
    ) -> None:
        """Test closing client with owned HTTP client."""
        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
        )

        # Create client to own
        await client._get_http_client()
        mock_client = client._http_client
        mock_client.aclose = AsyncMock()

        await client.close()

        mock_client.aclose.assert_called_once()
        assert client._http_client is None

    async def test_close_with_external_client(
        self,
        mock_config: CopilotOAuthConfig,
        mock_storage: CopilotOAuthStorage,
        mock_http_client: MagicMock,
    ) -> None:
        """Test closing client with external HTTP client."""
        mock_http_client.aclose = AsyncMock()

        client = CopilotOAuthClient(
            config=mock_config,
            storage=mock_storage,
            http_client=mock_http_client,
        )

        await client.close()

        # Should not close external client
        mock_http_client.aclose.assert_not_called()
