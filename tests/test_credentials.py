"""Tests for Claude credentials service."""

import json
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from claude_code_proxy.services.credentials import (
    AccountInfo,
    ClaudeCredentials,
    CredentialsService,
    OAuthToken,
    OrganizationInfo,
    UserProfile,
)


class TestOAuthToken:
    """Test OAuth token model."""

    def test_oauth_token_parsing(self):
        """Test parsing OAuth token from JSON data."""
        token_data = {
            "accessToken": "test-access-token",
            "refreshToken": "test-refresh-token",
            "expiresAt": 1751896667201,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }

        token = OAuthToken.model_validate(token_data)

        assert token.access_token == "test-access-token"
        assert token.refresh_token == "test-refresh-token"
        assert token.expires_at == 1751896667201
        assert token.scopes == ["user:inference", "user:profile"]
        assert token.subscription_type == "max"

    def test_token_expiry_check(self):
        """Test token expiry checking."""
        # Create a token that expires in 1 hour
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": future_ms,
            }
        )

        assert not token.is_expired

        # Create an expired token
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        expired_token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": past_ms,
            }
        )

        assert expired_token.is_expired

    def test_expires_at_datetime_conversion(self):
        """Test conversion of expires_at to datetime."""
        # Use a known timestamp
        timestamp_ms = 1751896667201
        expected_dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)

        token = OAuthToken.model_validate(
            {
                "accessToken": "test-token",
                "refreshToken": "refresh-token",
                "expiresAt": timestamp_ms,
            }
        )

        assert token.expires_at_datetime == expected_dt


class TestClaudeCredentials:
    """Test Claude credentials model."""

    def test_credentials_parsing(self):
        """Test parsing full credentials from JSON."""
        creds_data = {
            "claudeAiOauth": {
                "accessToken": "test-access-token",
                "refreshToken": "test-refresh-token",
                "expiresAt": 1751896667201,
                "scopes": ["user:inference"],
                "subscriptionType": "pro",
            }
        }

        credentials = ClaudeCredentials.model_validate(creds_data)

        assert credentials.claude_ai_oauth.access_token == "test-access-token"
        assert credentials.claude_ai_oauth.subscription_type == "pro"


class TestCredentialsService:
    """Test credentials service functionality."""

    def test_find_credentials_file_found(self):
        """Test finding credentials file when it exists."""
        # Just test that it returns a valid path when a file exists
        with (
            patch.object(Path, "exists", return_value=True),
            patch.object(Path, "is_file", return_value=True),
        ):
            result = CredentialsService.find_credentials_file()
            assert result is not None
            assert str(result).endswith("credentials.json")

    @patch("claude_code_proxy.services.credentials.Path.home")
    def test_find_credentials_file_not_found(self, mock_home):
        """Test when no credentials file exists."""
        mock_home.return_value = Path("/home/test")

        with patch("pathlib.Path.exists", return_value=False):
            result = CredentialsService.find_credentials_file()
            assert result is None

    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_load_credentials_success(self, mock_file_open, mock_find_file):
        """Test successful loading of credentials."""
        mock_find_file.return_value = Path("/home/test/.claude/credentials.json")

        mock_file_open.return_value.read.return_value = json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "test-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1751896667201,
                    "scopes": ["user:inference"],
                    "subscriptionType": "max",
                }
            }
        )

        credentials = CredentialsService.load_credentials()

        assert credentials is not None
        assert credentials.claude_ai_oauth.access_token == "test-token"
        assert credentials.claude_ai_oauth.subscription_type == "max"

    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    def test_load_credentials_file_not_found(self, mock_find_file):
        """Test loading when credentials file doesn't exist."""
        mock_find_file.return_value = None

        credentials = CredentialsService.load_credentials()
        assert credentials is None

    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    @patch("pathlib.Path.open", new_callable=mock_open)
    def test_load_credentials_invalid_json(self, mock_file_open, mock_find_file):
        """Test loading with invalid JSON."""
        mock_find_file.return_value = Path("/home/test/.claude/credentials.json")
        mock_file_open.return_value.read.return_value = "invalid json"

        credentials = CredentialsService.load_credentials()
        assert credentials is None

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_validate_credentials_valid(self, mock_load):
        """Test validation with valid credentials."""
        # Create non-expired token
        future_time = datetime.now(UTC) + timedelta(days=7)
        future_ms = int(future_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "test-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": future_ms,
                    "scopes": ["user:inference"],
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds

        result = CredentialsService.validate_credentials()

        assert result["valid"] is True
        assert result["expired"] is False
        assert result["subscription_type"] == "max"
        assert result["scopes"] == ["user:inference"]
        assert "expires_at" in result

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_validate_credentials_expired(self, mock_load):
        """Test validation with expired credentials."""
        # Create expired token
        past_time = datetime.now(UTC) - timedelta(days=1)
        past_ms = int(past_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "test-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": past_ms,
                    "subscriptionType": "pro",
                }
            }
        )
        mock_load.return_value = mock_creds

        result = CredentialsService.validate_credentials()

        assert result["valid"] is True
        assert result["expired"] is True
        assert result["subscription_type"] == "pro"

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_validate_credentials_not_found(self, mock_load):
        """Test validation when no credentials found."""
        mock_load.return_value = None

        result = CredentialsService.validate_credentials()

        assert result["valid"] is False
        assert "error" in result
        error_msg = result.get("error", "")
        assert isinstance(error_msg, str)
        assert "No credentials file found" in error_msg

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_get_access_token_valid(self, mock_load):
        """Test getting access token when valid."""
        # Create non-expired token
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "valid-access-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": future_ms,
                }
            }
        )
        mock_load.return_value = mock_creds

        token = CredentialsService.get_access_token()
        assert token == "valid-access-token"

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_get_access_token_expired(self, mock_load):
        """Test getting access token when expired."""
        # Create expired token
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": past_ms,
                }
            }
        )
        mock_load.return_value = mock_creds

        token = CredentialsService.get_access_token()
        assert token is None

    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    def test_get_access_token_no_credentials(self, mock_load):
        """Test getting access token when no credentials."""
        mock_load.return_value = None

        token = CredentialsService.get_access_token()
        assert token is None

    def test_find_credentials_file_with_custom_paths(self):
        """Test finding credentials file with custom paths."""
        custom_paths = [
            Path("/custom/path1/.credentials.json"),
            Path("/custom/path2/.credentials.json"),
        ]

        # Mock exists to return True only for the second path
        def mock_exists(self):
            return str(self) == "/custom/path2/.credentials.json"

        with (
            patch.object(Path, "exists", mock_exists),
            patch.object(Path, "is_file", return_value=True),
        ):
            result = CredentialsService.find_credentials_file(custom_paths)
            assert result == Path("/custom/path2/.credentials.json")

    def test_load_credentials_with_custom_paths(self):
        """Test loading credentials with custom paths."""
        custom_paths = [Path("/custom/.credentials.json")]

        with (
            patch(
                "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
            ) as mock_find,
            patch("pathlib.Path.open", new_callable=mock_open) as mock_file_open,
        ):
            mock_find.return_value = Path("/custom/.credentials.json")
            mock_file_open.return_value.read.return_value = json.dumps(
                {
                    "claudeAiOauth": {
                        "accessToken": "custom-token",
                        "refreshToken": "custom-refresh",
                        "expiresAt": 1751896667201,
                        "subscriptionType": "max",
                    }
                }
            )

            credentials = CredentialsService.load_credentials(custom_paths)

            mock_find.assert_called_once_with(custom_paths)
            assert credentials is not None
            assert credentials.claude_ai_oauth.access_token == "custom-token"

    def test_validate_credentials_with_custom_paths(self):
        """Test validation with custom paths."""
        custom_paths = [Path("/custom/.credentials.json")]

        with patch(
            "claude_code_proxy.services.credentials.CredentialsService.load_credentials"
        ) as mock_load:
            future_time = datetime.now(UTC) + timedelta(days=1)
            future_ms = int(future_time.timestamp() * 1000)

            mock_creds = ClaudeCredentials.model_validate(
                {
                    "claudeAiOauth": {
                        "accessToken": "test-token",
                        "refreshToken": "refresh-token",
                        "expiresAt": future_ms,
                        "subscriptionType": "max",
                    }
                }
            )
            mock_load.return_value = mock_creds

            result = CredentialsService.validate_credentials(custom_paths)

            mock_load.assert_called_once_with(custom_paths)
            assert result["valid"] is True
            assert result["expired"] is False

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    def test_save_credentials_success(self, mock_find_file, mock_file_open):
        """Test successful saving of credentials."""
        mock_find_file.return_value = Path("/home/test/.claude/credentials.json")

        credentials = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "new-token",
                    "refreshToken": "new-refresh",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )

        result = CredentialsService.save_credentials(credentials)

        assert result is True
        mock_find_file.assert_called_once_with(None)
        mock_file_open.assert_called_once()

    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    def test_save_credentials_file_not_found(self, mock_find_file):
        """Test saving when credentials file doesn't exist."""
        mock_find_file.return_value = None

        credentials = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "new-token",
                    "refreshToken": "new-refresh",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )

        result = CredentialsService.save_credentials(credentials)
        assert result is False

    @patch("pathlib.Path.open", new_callable=mock_open)
    @patch(
        "claude_code_proxy.services.credentials.CredentialsService.find_credentials_file"
    )
    def test_save_credentials_with_custom_paths(self, mock_find_file, mock_file_open):
        """Test saving credentials with custom paths."""
        custom_paths = [Path("/custom/.credentials.json")]
        mock_find_file.return_value = Path("/custom/.credentials.json")

        credentials = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "custom-token",
                    "refreshToken": "custom-refresh",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )

        result = CredentialsService.save_credentials(credentials, custom_paths)

        assert result is True
        mock_find_file.assert_called_once_with(custom_paths)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("claude_code_proxy.services.credentials.CredentialsService.save_credentials")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_refresh_token_success(self, mock_load, mock_save, mock_client):
        """Test successful token refresh."""
        # Mock existing credentials
        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "old-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds
        mock_save.return_value = True

        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 28800,  # 8 hours
            "scope": "user:inference user:profile",
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        # Test the refresh
        new_token, updated_creds = await CredentialsService.refresh_token()

        assert new_token == "new-access-token"
        assert updated_creds is not None
        assert updated_creds.claude_ai_oauth.access_token == "new-access-token"
        assert updated_creds.claude_ai_oauth.refresh_token == "new-refresh-token"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_refresh_token_no_credentials(self, mock_load):
        """Test token refresh when no credentials exist."""
        mock_load.return_value = None

        result = await CredentialsService.refresh_token()
        assert result == (None, None)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_refresh_token_http_error(self, mock_load, mock_client):
        """Test token refresh with HTTP error."""
        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "old-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds

        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        result = await CredentialsService.refresh_token()
        assert result == (None, None)

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.credentials.CredentialsService.refresh_token")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_get_access_token_with_refresh_valid_token(
        self, mock_load, mock_refresh
    ):
        """Test get_access_token_with_refresh when token is valid."""
        # Create non-expired token
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "valid-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": future_ms,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds

        token = await CredentialsService.get_access_token_with_refresh()

        assert token == "valid-token"
        mock_refresh.assert_not_called()

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.credentials.CredentialsService.refresh_token")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_get_access_token_with_refresh_expired_token(
        self, mock_load, mock_refresh
    ):
        """Test get_access_token_with_refresh when token is expired."""
        # Create expired token
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": past_ms,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds
        mock_refresh.return_value = ("new-token", mock_creds)

        token = await CredentialsService.get_access_token_with_refresh()

        assert token == "new-token"
        mock_refresh.assert_called_once()

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.credentials.CredentialsService.refresh_token")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_get_access_token_with_refresh_custom_paths(
        self, mock_load, mock_refresh
    ):
        """Test get_access_token_with_refresh with custom paths."""
        custom_paths = [Path("/custom/.credentials.json")]

        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": past_ms,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds
        mock_refresh.return_value = ("refreshed-token", mock_creds)

        token = await CredentialsService.get_access_token_with_refresh(custom_paths)

        assert token == "refreshed-token"
        mock_load.assert_called_with(custom_paths)
        mock_refresh.assert_called_with(custom_paths)

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("claude_code_proxy.services.credentials.CredentialsService.save_credentials")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_fetch_user_profile_success(self, mock_load, mock_save, mock_client):
        """Test successful user profile fetch with token refresh."""
        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "access-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds
        mock_save.return_value = True

        # Mock HTTP response with profile data
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 28800,
            "scope": "user:inference user:profile",
            "organization": {
                "uuid": "org-123",
                "name": "Test Organization",
            },
            "account": {
                "uuid": "acc-456",
                "email_address": "test@example.com",
            },
        }

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        profile = await CredentialsService.fetch_user_profile("access-token")

        assert profile is not None
        assert profile.organization is not None
        assert profile.organization.name == "Test Organization"
        assert profile.account is not None
        assert profile.account.email_address == "test@example.com"
        mock_save.assert_called_once()

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_fetch_user_profile_no_credentials(self, mock_load):
        """Test user profile fetch when no credentials exist."""
        mock_load.return_value = None

        profile = await CredentialsService.fetch_user_profile("access-token")
        assert profile is None

    @pytest.mark.asyncio
    @patch("httpx.AsyncClient")
    @patch("claude_code_proxy.services.credentials.CredentialsService.load_credentials")
    async def test_fetch_user_profile_http_error(self, mock_load, mock_client):
        """Test user profile fetch with HTTP error."""
        mock_creds = ClaudeCredentials.model_validate(
            {
                "claudeAiOauth": {
                    "accessToken": "access-token",
                    "refreshToken": "refresh-token",
                    "expiresAt": 1751896667201,
                    "subscriptionType": "max",
                }
            }
        )
        mock_load.return_value = mock_creds

        # Mock HTTP error response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server Error"

        mock_client_instance = AsyncMock()
        mock_client_instance.__aenter__.return_value = mock_client_instance
        mock_client_instance.post.return_value = mock_response
        mock_client.return_value = mock_client_instance

        profile = await CredentialsService.fetch_user_profile("access-token")
        assert profile is None


class TestUserProfile:
    """Test user profile models."""

    def test_organization_info_creation(self):
        """Test creating organization info."""
        org = OrganizationInfo(uuid="org-123", name="Test Org")
        assert org.uuid == "org-123"
        assert org.name == "Test Org"

    def test_account_info_creation(self):
        """Test creating account info."""
        account = AccountInfo(uuid="acc-456", email_address="test@example.com")
        assert account.uuid == "acc-456"
        assert account.email_address == "test@example.com"

    def test_user_profile_creation(self):
        """Test creating user profile."""
        org = OrganizationInfo(uuid="org-123", name="Test Org")
        account = AccountInfo(uuid="acc-456", email_address="test@example.com")

        profile = UserProfile(organization=org, account=account)
        assert profile.organization == org
        assert profile.account == account

    def test_user_profile_optional_fields(self):
        """Test user profile with optional fields."""
        profile = UserProfile()
        assert profile.organization is None
        assert profile.account is None
