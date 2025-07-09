"""Tests for keyring storage functionality."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ccproxy.services.credentials import JsonFileStorage
from ccproxy.services.credentials.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.services.credentials.models import ClaudeCredentials, OAuthToken


class TestKeyringStorage:
    """Test keyring storage functionality in JsonFileStorage."""

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
                subscriptionType="pro",
            )
        )

    @pytest.fixture
    def credentials_json(self, valid_credentials):
        """Get JSON representation of valid credentials."""
        return json.dumps(valid_credentials.model_dump(by_alias=True))

    @pytest.mark.asyncio
    async def test_keyring_available_load_from_keyring(
        self, tmp_path, valid_credentials, credentials_json
    ):
        """Test loading credentials from keyring when available."""
        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
                mock_keyring.get_password.return_value = credentials_json

                storage = JsonFileStorage(tmp_path / "credentials.json")

                # Should load from keyring
                result = await storage.load()

                assert result is not None
                assert result.claude_ai_oauth.access_token == "test-access-token"
                mock_keyring.get_password.assert_called_once_with(
                    "ccproxy", "credentials"
                )

    @pytest.mark.asyncio
    async def test_keyring_available_fallback_to_file(
        self, tmp_path, valid_credentials
    ):
        """Test falling back to file when keyring fails."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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
                assert json.loads(
                    creds_file.read_text()
                ) == valid_credentials.model_dump(by_alias=True)
                # Check file permissions
                assert oct(creds_file.stat().st_mode)[-3:] == "600"

    @pytest.mark.asyncio
    async def test_keyring_available_save_continues_on_keyring_failure(
        self, tmp_path, valid_credentials
    ):
        """Test that save continues to file even if keyring fails."""
        creds_file = tmp_path / "credentials.json"

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
                # Keyring save fails
                mock_keyring.set_password.side_effect = Exception("Keyring error")

                storage = JsonFileStorage(creds_file)

                # Save should still succeed
                result = await storage.save(valid_credentials)

                assert result is True
                # File should be saved
                assert creds_file.exists()
                assert json.loads(
                    creds_file.read_text()
                ) == valid_credentials.model_dump(by_alias=True)

    @pytest.mark.asyncio
    async def test_keyring_not_available_file_only(self, tmp_path, valid_credentials):
        """Test behavior when keyring is not available."""
        creds_file = tmp_path / "credentials.json"

        with patch(
            "ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", False
        ):
            storage = JsonFileStorage(creds_file)

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

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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

    def test_get_location_with_keyring(self, tmp_path):
        """Test location string when keyring is available."""
        creds_file = tmp_path / "credentials.json"

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):
            storage = JsonFileStorage(creds_file)
            location = storage.get_location()

            assert str(creds_file) in location
            assert "with keyring support" in location

    def test_get_location_without_keyring(self, tmp_path):
        """Test location string when keyring is not available."""
        creds_file = tmp_path / "credentials.json"

        with patch(
            "ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", False
        ):
            storage = JsonFileStorage(creds_file)
            location = storage.get_location()

            assert location == str(creds_file)
            assert "keyring" not in location

    @pytest.mark.asyncio
    async def test_keyring_no_credentials_in_keyring(self, tmp_path, valid_credentials):
        """Test when keyring has no credentials but file does."""
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps(valid_credentials.model_dump(by_alias=True)))

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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

        with patch("ccproxy.services.credentials.json_storage.KEYRING_AVAILABLE", True):  # noqa: SIM117
            with patch(
                "ccproxy.services.credentials.json_storage.keyring"
            ) as mock_keyring:
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
