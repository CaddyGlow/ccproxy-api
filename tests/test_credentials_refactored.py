"""Tests for the refactored credentials system."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from ccproxy.services.credentials import (
    CredentialsConfig,
    CredentialsManager,
    JsonFileStorage,
    OAuthClient,
    OAuthConfig,
)
from ccproxy.auth.exceptions import (
    CredentialsExpiredError,
    CredentialsInvalidError,
    CredentialsNotFoundError,
    CredentialsStorageError,
    OAuthLoginError,
    OAuthTokenRefreshError,
)
from ccproxy.auth.models import (
    ClaudeCredentials,
    OAuthToken,
)


class TestJsonFileStorage:
    """Test JSON file storage backend."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create a storage instance with a temporary file."""
        return JsonFileStorage(tmp_path / "credentials.json")

    @pytest.fixture
    def valid_credentials(self):
        """Create valid test credentials."""
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="test-access-token",
                refreshToken="test-refresh-token",
                expiresAt=future_ms,
                scopes=["user:inference", "user:profile"],
                subscriptionType="max",
            )
        )

    @pytest.mark.asyncio
    async def test_save_and_load(self, storage, valid_credentials):
        """Test saving and loading credentials."""
        # Initially should not exist
        assert not await storage.exists()

        # Save credentials
        assert await storage.save(valid_credentials)
        assert await storage.exists()

        # Load credentials
        loaded = await storage.load()
        assert loaded is not None
        assert loaded.claude_ai_oauth.access_token == "test-access-token"
        assert loaded.claude_ai_oauth.refresh_token == "test-refresh-token"

    @pytest.mark.asyncio
    async def test_load_nonexistent(self, storage, mock_empty_keyring):
        """Test loading from non-existent file."""
        result = await storage.load()
        assert result is None

    @pytest.mark.asyncio
    async def test_load_invalid_json(self, storage, mock_empty_keyring):
        """Test loading invalid JSON."""
        # Create invalid JSON file
        storage.file_path.write_text("invalid json{")

        with pytest.raises(CredentialsInvalidError):
            await storage.load()

    @pytest.mark.asyncio
    async def test_delete(self, storage, valid_credentials):
        """Test deleting credentials."""
        # Save then delete
        await storage.save(valid_credentials)
        assert await storage.exists()

        assert await storage.delete()
        assert not await storage.exists()

        # Delete non-existent should return False
        assert not await storage.delete()

    def test_get_location(self, tmp_path):
        """Test getting storage location."""
        # Create storage (keyring is enabled by default)
        storage = JsonFileStorage(tmp_path / "credentials.json")
        location = storage.get_location()
        # Should include keyring support by default
        assert str(storage.file_path) in location
        assert "with keyring support" in location


    # Keyring-specific tests (merged from test_keyring_storage.py)
    
    @pytest.fixture
    def credentials_json(self, valid_credentials):
        """Get JSON representation of valid credentials."""
        return json.dumps(valid_credentials.model_dump(by_alias=True))

    @pytest.mark.asyncio
    async def test_keyring_available_load_from_keyring(
        self, tmp_path, valid_credentials, credentials_json
    ):
        """Test loading credentials from keyring when available."""
        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            mock_keyring.get_password.return_value = credentials_json

            storage = JsonFileStorage(tmp_path / "credentials.json")

            # Should load from keyring
            result = await storage.load()

            assert result is not None
            assert result.claude_ai_oauth.access_token == "test-access-token"
            mock_keyring.get_password.assert_called_once_with("ccproxy", "credentials")

    @pytest.mark.asyncio
    async def test_keyring_available_fallback_to_file(
        self, tmp_path, valid_credentials
    ):
        """Test falling back to file when keyring fails."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            # Keyring fails
            mock_keyring.get_password.side_effect = Exception("Keyring error")

            storage = JsonFileStorage(creds_file)

            # Should load from file
            result = await storage.load()

            assert result is not None
            assert result.claude_ai_oauth.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_keyring_available_save_to_both(
        self, tmp_path, valid_credentials, credentials_json
    ):
        """Test saving to both keyring and file when keyring is available."""
        creds_file = tmp_path / "credentials.json"

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            storage = JsonFileStorage(creds_file)

            # Save credentials
            result = await storage.save(valid_credentials)

            assert result is True
            # Should save to keyring
            mock_keyring.set_password.assert_called_once_with(
                "ccproxy", "credentials", credentials_json
            )
            # Should also save to file
            assert creds_file.exists()
            assert json.loads(creds_file.read_text()) == valid_credentials.model_dump(
                by_alias=True
            )
            # Check file permissions
            assert oct(creds_file.stat().st_mode)[-3:] == "600"

    @pytest.mark.asyncio
    async def test_keyring_available_save_continues_on_keyring_failure(
        self, tmp_path, valid_credentials
    ):
        """Test that save continues to file even if keyring fails."""
        creds_file = tmp_path / "credentials.json"

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            # Keyring save fails
            mock_keyring.set_password.side_effect = Exception("Keyring error")

            storage = JsonFileStorage(creds_file)

            # Save should still succeed
            result = await storage.save(valid_credentials)

            assert result is True
            # File should be saved
            assert creds_file.exists()
            assert json.loads(creds_file.read_text()) == valid_credentials.model_dump(
                by_alias=True
            )

    @pytest.mark.asyncio
    async def test_keyring_not_available_file_only(self, tmp_path, valid_credentials):
        """Test behavior when keyring is not available."""
        creds_file = tmp_path / "credentials.json"

        # Test when keyring functionality is disabled by setting _keyring_available to False
        storage = JsonFileStorage(creds_file)
        storage._keyring_available = False

        # Save credentials
        await storage.save(valid_credentials)

        # Load credentials
        result = await storage.load()

        assert result is not None
        assert result.claude_ai_oauth.access_token == "test-access-token"
        assert creds_file.exists()

    @pytest.mark.asyncio
    async def test_keyring_available_delete_from_both(
        self, tmp_path, valid_credentials
    ):
        """Test deleting from both keyring and file."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            storage = JsonFileStorage(creds_file)

            # Delete credentials
            result = await storage.delete()

            assert result is True
            # Should delete from keyring
            mock_keyring.delete_password.assert_called_once_with(
                "ccproxy", "credentials"
            )
            # Should also delete file
            assert not creds_file.exists()

    @pytest.mark.asyncio
    async def test_keyring_available_delete_continues_on_keyring_failure(
        self, tmp_path
    ):
        """Test that delete continues to file even if keyring fails."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text("{}")

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            # Keyring delete fails
            mock_keyring.delete_password.side_effect = Exception(
                "No such keyring entry"
            )

            storage = JsonFileStorage(creds_file)

            # Delete should still succeed
            result = await storage.delete()

            assert result is True
            # File should be deleted
            assert not creds_file.exists()

    def test_get_location_with_keyring_explicit(self, tmp_path):
        """Test location string when keyring is explicitly available."""
        creds_file = tmp_path / "credentials.json"

        storage = JsonFileStorage(creds_file)
        storage._keyring_available = True
        location = storage.get_location()

        assert str(creds_file) in location
        assert "with keyring support" in location

    def test_get_location_without_keyring_explicit(self, tmp_path):
        """Test location string when keyring is explicitly not available."""
        creds_file = tmp_path / "credentials.json"

        # Test when keyring functionality is disabled by setting _keyring_available to False
        storage = JsonFileStorage(creds_file)
        storage._keyring_available = False
        location = storage.get_location()

        assert location == str(creds_file)
        assert "keyring" not in location

    @pytest.mark.asyncio
    async def test_keyring_no_credentials_in_keyring(self, tmp_path, valid_credentials):
        """Test when keyring has no credentials but file does."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            # Keyring returns None (no credentials)
            mock_keyring.get_password.return_value = None

            storage = JsonFileStorage(creds_file)

            # Should load from file
            result = await storage.load()

            assert result is not None
            assert result.claude_ai_oauth.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_keyring_invalid_json_in_keyring(self, tmp_path, valid_credentials):
        """Test handling invalid JSON from keyring."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.keyring") as mock_keyring:
            # Keyring returns invalid JSON
            mock_keyring.get_password.return_value = "invalid json"

            storage = JsonFileStorage(creds_file)

            # Should fall back to file
            result = await storage.load()

            assert result is not None
            assert result.claude_ai_oauth.access_token == "test-access-token"


class TestCredentialStoragePaths:
    """Test credential storage path configuration."""

    def test_default_storage_paths(self):
        """Test that default storage paths include the app config directory."""
        from ccproxy.services.credentials.config import _get_default_storage_paths

        # Ensure we're not in test mode for this test
        with patch.dict("os.environ", {"CCPROXY_TEST_MODE": ""}, clear=True):
            paths = _get_default_storage_paths()

            # Only app config path should be present now
            assert paths == ["~/.config/ccproxy/credentials.json"]
            # Legacy paths have been removed to avoid sharing issues with Claude Code
            assert "~/.claude/.credentials.json" not in paths
            assert "~/.config/claude/.credentials.json" not in paths

    def test_test_mode_storage_paths(self):
        """Test storage paths in test mode."""
        from ccproxy.services.credentials.config import _get_default_storage_paths

        with patch.dict("os.environ", {"CCPROXY_TEST_MODE": "true"}):
            paths = _get_default_storage_paths()

            assert "/tmp/ccproxy-test/.claude/.credentials.json" in paths
            assert "/tmp/ccproxy-test/.config/claude/.credentials.json" in paths
            assert "~/.config/ccproxy/credentials.json" not in paths

class TestOAuthClient:
    """Test OAuth client."""

    @pytest.fixture
    def oauth_config(self):
        """Create test OAuth configuration."""
        return OAuthConfig(
            callback_timeout=60,  # Minimum allowed timeout
            callback_port=54546,  # Different port to avoid conflicts
        )

    @pytest.fixture
    def oauth_client(self, oauth_config):
        """Create OAuth client with test config."""
        return OAuthClient(config=oauth_config)

    @pytest.mark.asyncio
    async def test_refresh_token_success(self, oauth_client):
        """Test successful token refresh."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "new-access-token",
            "refresh_token": "new-refresh-token",
            "expires_in": 3600,
            "scope": "user:inference user:profile",
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        oauth_client._http_client = mock_client

        result = await oauth_client.refresh_token("old-refresh-token")

        assert result.access_token == "new-access-token"
        assert result.refresh_token == "new-refresh-token"
        assert not result.is_expired

    @pytest.mark.asyncio
    async def test_refresh_token_failure(self, oauth_client):
        """Test failed token refresh."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        oauth_client._http_client = mock_client

        with pytest.raises(OAuthTokenRefreshError):
            await oauth_client.refresh_token("bad-refresh-token")


class TestCredentialsManager:
    """Test credentials manager."""

    @pytest.fixture
    def temp_credentials_file(self, tmp_path):
        """Create a temporary credentials file path."""
        return tmp_path / "test_credentials.json"

    @pytest.fixture
    def config(self, temp_credentials_file):
        """Create test configuration."""
        return CredentialsConfig(
            storage_paths=[str(temp_credentials_file)],
            auto_refresh=True,
            refresh_buffer_seconds=300,
        )

    @pytest.fixture
    def manager(self, config, temp_credentials_file):
        """Create credentials manager with test config."""
        storage = JsonFileStorage(temp_credentials_file)
        return CredentialsManager(config=config, storage=storage)

    @pytest.fixture
    def valid_credentials(self):
        """Create valid test credentials."""
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="test-access-token",
                refreshToken="test-refresh-token",
                expiresAt=future_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

    @pytest.fixture
    def expired_credentials(self):
        """Create expired test credentials."""
        past_time = datetime.now(UTC) - timedelta(hours=1)
        past_ms = int(past_time.timestamp() * 1000)

        return ClaudeCredentials(
            claudeAiOauth=OAuthToken(
                accessToken="expired-token",
                refreshToken="refresh-token",
                expiresAt=past_ms,
                scopes=["user:inference"],
                subscriptionType="pro",
            )
        )

    @pytest.mark.asyncio
    async def test_save_and_load(self, manager, valid_credentials):
        """Test saving and loading credentials."""
        # Save
        assert await manager.save(valid_credentials)

        # Load
        loaded = await manager.load()
        assert loaded is not None
        assert loaded.claude_ai_oauth.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_get_valid_credentials(self, manager, valid_credentials):
        """Test getting valid credentials."""
        await manager.save(valid_credentials)

        result = await manager.get_valid_credentials()
        assert result.claude_ai_oauth.access_token == "test-access-token"

    @pytest.mark.asyncio
    async def test_get_valid_credentials_not_found(self, manager, mock_empty_keyring):
        """Test getting credentials when none exist."""
        with pytest.raises(CredentialsNotFoundError):
            await manager.get_valid_credentials()

    @pytest.mark.asyncio
    async def test_get_valid_credentials_refresh(self, manager, expired_credentials):
        """Test automatic token refresh."""
        await manager.save(expired_credentials)

        # Mock the OAuth client's refresh method
        mock_oauth_client = MagicMock()
        future_time = datetime.now(UTC) + timedelta(hours=1)
        future_ms = int(future_time.timestamp() * 1000)

        new_token = OAuthToken(
            accessToken="refreshed-token",
            refreshToken="new-refresh-token",
            expiresAt=future_ms,
            scopes=["user:inference"],
            subscriptionType="pro",
        )
        mock_oauth_client.refresh_token = AsyncMock(return_value=new_token)
        manager._oauth_client = mock_oauth_client

        result = await manager.get_valid_credentials()
        assert result.claude_ai_oauth.access_token == "refreshed-token"

    @pytest.mark.asyncio
    async def test_validate(self, manager, valid_credentials):
        """Test credentials validation."""
        await manager.save(valid_credentials)

        result = await manager.validate()
        assert result["valid"] is True
        assert result["expired"] is False
        assert result["subscription_type"] == "pro"

    @pytest.mark.asyncio
    async def test_validate_no_credentials(self, manager, mock_empty_keyring):
        """Test validation when no credentials exist."""
        result = await manager.validate()
        assert result["valid"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_logout(self, manager, valid_credentials):
        """Test logout (delete credentials)."""
        await manager.save(valid_credentials)
        assert await manager.storage.exists()

        assert await manager.logout()
        assert not await manager.storage.exists()


class TestCredentialsConfig:
    """Test credentials configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CredentialsConfig()

        # In test mode, paths should be test paths
        assert config.storage_paths == [
            "/tmp/ccproxy-test/.claude/.credentials.json",
            "/tmp/ccproxy-test/.config/claude/.credentials.json",
        ]
        assert config.auto_refresh is True
        assert config.refresh_buffer_seconds == 300

        # OAuth config defaults
        assert config.oauth.client_id == "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
        assert config.oauth.callback_port == 54545

    def test_custom_config(self):
        """Test custom configuration."""
        config = CredentialsConfig(
            storage_paths=["/custom/path.json"],
            auto_refresh=False,
            oauth=OAuthConfig(callback_port=12345),
        )

        assert config.storage_paths == ["/custom/path.json"]
        assert config.auto_refresh is False
        assert config.oauth.callback_port == 12345
