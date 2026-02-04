"""Unit tests for ClaudeOAuthStorage including keychain support."""

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr

from ccproxy.plugins.oauth_claude.models import (
    ClaudeCredentials,
    ClaudeOAuthToken,
)
from ccproxy.plugins.oauth_claude.storage import (
    KEYCHAIN_ACCOUNT,
    KEYCHAIN_SERVICE,
    ClaudeOAuthStorage,
    _is_keyring_available,
    _read_from_keychain,
)


class TestKeychainAvailability:
    """Test keyring availability detection."""

    def test_is_keyring_available_when_installed(self) -> None:
        """Test detection when keyring is installed."""
        # Force reimport to pick up the mock
        with (
            patch.dict("sys.modules", {"keyring": MagicMock()}),
            patch("builtins.__import__", return_value=MagicMock()),
        ):
            assert _is_keyring_available() is True

    def test_is_keyring_available_when_not_installed(self) -> None:
        """Test detection when keyring is not installed."""
        with patch("builtins.__import__", side_effect=ImportError("No module")):
            assert _is_keyring_available() is False


class TestReadFromKeychain:
    """Test keychain reading functionality."""

    @pytest.fixture
    def mock_credentials_data(self) -> dict[str, Any]:
        """Create mock credentials data as stored in keychain."""
        return {
            "claudeAiOauth": {
                "accessToken": "test_access_token",
                "refreshToken": "test_refresh_token",
                "expiresAt": 1234567890000,
                "scopes": ["user:read"],
                "subscriptionType": "pro",
            }
        }

    @pytest.mark.asyncio
    async def test_read_from_keychain_success(
        self, mock_credentials_data: dict[str, Any]
    ) -> None:
        """Test successful read from keychain."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = json.dumps(mock_credentials_data)

        with (
            patch.dict("sys.modules", {"keyring": mock_keyring}),
            patch(
                "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
                return_value=True,
            ),
            patch(
                "ccproxy.plugins.oauth_claude.storage.asyncio.to_thread",
                side_effect=lambda f: f(),
            ),
        ):
            # Patch the import inside the function
            import sys

            sys.modules["keyring"] = mock_keyring

            result = await _read_from_keychain()

            assert result is not None
            assert result == mock_credentials_data
            mock_keyring.get_password.assert_called_once_with(
                KEYCHAIN_SERVICE, KEYCHAIN_ACCOUNT
            )

    @pytest.mark.asyncio
    async def test_read_from_keychain_not_found(self) -> None:
        """Test when credentials not found in keychain."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = None

        with (
            patch(
                "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
                return_value=True,
            ),
            patch(
                "ccproxy.plugins.oauth_claude.storage.asyncio.to_thread",
                side_effect=lambda f: f(),
            ),
        ):
            import sys

            sys.modules["keyring"] = mock_keyring

            result = await _read_from_keychain()

            assert result is None

    @pytest.mark.asyncio
    async def test_read_from_keychain_keyring_unavailable(self) -> None:
        """Test when keyring library is not available."""
        with patch(
            "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
            return_value=False,
        ):
            result = await _read_from_keychain()

            assert result is None

    @pytest.mark.asyncio
    async def test_read_from_keychain_invalid_json(self) -> None:
        """Test when keychain contains invalid JSON."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = "not valid json{"

        with (
            patch(
                "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
                return_value=True,
            ),
            patch(
                "ccproxy.plugins.oauth_claude.storage.asyncio.to_thread",
                side_effect=lambda f: f(),
            ),
        ):
            import sys

            sys.modules["keyring"] = mock_keyring

            result = await _read_from_keychain()

            assert result is None

    @pytest.mark.asyncio
    async def test_read_from_keychain_non_dict_json(self) -> None:
        """Test when keychain contains valid JSON but not a dict."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.return_value = '["list", "not", "dict"]'

        with (
            patch(
                "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
                return_value=True,
            ),
            patch(
                "ccproxy.plugins.oauth_claude.storage.asyncio.to_thread",
                side_effect=lambda f: f(),
            ),
        ):
            import sys

            sys.modules["keyring"] = mock_keyring

            result = await _read_from_keychain()

            assert result is None

    @pytest.mark.asyncio
    async def test_read_from_keychain_exception(self) -> None:
        """Test when keyring raises an exception."""
        mock_keyring = MagicMock()
        mock_keyring.get_password.side_effect = Exception("Keychain locked")

        with (
            patch(
                "ccproxy.plugins.oauth_claude.storage._is_keyring_available",
                return_value=True,
            ),
            patch(
                "ccproxy.plugins.oauth_claude.storage.asyncio.to_thread",
                side_effect=lambda f: f(),
            ),
        ):
            import sys

            sys.modules["keyring"] = mock_keyring

            result = await _read_from_keychain()

            assert result is None


class TestClaudeOAuthStorageKeychain:
    """Test ClaudeOAuthStorage keychain fallback."""

    @pytest.fixture
    def temp_storage_path(self, tmp_path: Path) -> Iterator[Path]:
        """Create temporary storage path."""
        yield tmp_path / ".claude" / ".credentials.json"

    @pytest.fixture
    def mock_credentials(self) -> ClaudeCredentials:
        """Create mock credentials."""
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_access"),
            refreshToken=SecretStr("test_refresh"),
            expiresAt=1234567890000,
            subscriptionType="pro",
        )
        return ClaudeCredentials(claudeAiOauth=oauth)

    @pytest.mark.asyncio
    async def test_load_from_file_first(
        self, temp_storage_path: Path, mock_credentials: ClaudeCredentials
    ) -> None:
        """Test that file is tried before keychain."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)

        # Save credentials to file
        await storage.save(mock_credentials)

        # Mock keychain to verify it's not called when file exists
        with patch(
            "ccproxy.plugins.oauth_claude.storage._read_from_keychain"
        ) as mock_keychain:
            loaded = await storage.load()

            assert loaded is not None
            assert (
                loaded.claude_ai_oauth.access_token.get_secret_value() == "test_access"
            )
            mock_keychain.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_falls_back_to_keychain(self, temp_storage_path: Path) -> None:
        """Test fallback to keychain when file doesn't exist."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)

        keychain_data = {
            "claudeAiOauth": {
                "accessToken": "keychain_token",
                "refreshToken": "keychain_refresh",
                "expiresAt": 1234567890000,
            }
        }

        with patch(
            "ccproxy.plugins.oauth_claude.storage._read_from_keychain",
            return_value=keychain_data,
        ):
            loaded = await storage.load()

            assert loaded is not None
            assert (
                loaded.claude_ai_oauth.access_token.get_secret_value()
                == "keychain_token"
            )

    @pytest.mark.asyncio
    async def test_load_returns_none_when_both_fail(
        self, temp_storage_path: Path
    ) -> None:
        """Test returns None when both file and keychain fail."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)

        with patch(
            "ccproxy.plugins.oauth_claude.storage._read_from_keychain",
            return_value=None,
        ):
            loaded = await storage.load()

            assert loaded is None


class TestClaudeOAuthStorageBasic:
    """Test basic ClaudeOAuthStorage operations."""

    @pytest.fixture
    def temp_storage_path(self, tmp_path: Path) -> Iterator[Path]:
        """Create temporary storage path."""
        yield tmp_path / ".claude" / ".credentials.json"

    @pytest.fixture
    def mock_credentials(self) -> ClaudeCredentials:
        """Create mock credentials."""
        oauth = ClaudeOAuthToken(
            accessToken=SecretStr("test_access"),
            refreshToken=SecretStr("test_refresh"),
            expiresAt=1234567890000,
            subscriptionType=None,
        )
        return ClaudeCredentials(claudeAiOauth=oauth)

    def test_default_storage_path(self) -> None:
        """Test default storage path is ~/.claude/.credentials.json."""
        storage = ClaudeOAuthStorage()
        expected = Path.home() / ".claude" / ".credentials.json"
        assert storage.file_path == expected

    def test_custom_storage_path(self, temp_storage_path: Path) -> None:
        """Test custom storage path."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)
        assert storage.file_path == temp_storage_path

    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(
        self, temp_storage_path: Path, mock_credentials: ClaudeCredentials
    ) -> None:
        """Test save and load round trip."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)

        # Save
        result = await storage.save(mock_credentials)
        assert result is True
        assert temp_storage_path.exists()

        # Load (mock keychain to ensure we're testing file loading)
        with patch(
            "ccproxy.plugins.oauth_claude.storage._read_from_keychain",
            return_value=None,
        ):
            loaded = await storage.load()

        assert loaded is not None
        assert loaded.claude_ai_oauth.access_token.get_secret_value() == "test_access"
        assert loaded.claude_ai_oauth.refresh_token.get_secret_value() == "test_refresh"
        assert loaded.claude_ai_oauth.expires_at == 1234567890000

    @pytest.mark.asyncio
    async def test_load_nonexistent_file_no_keychain(
        self, temp_storage_path: Path
    ) -> None:
        """Test loading when file doesn't exist and keychain unavailable."""
        storage = ClaudeOAuthStorage(storage_path=temp_storage_path)

        with patch(
            "ccproxy.plugins.oauth_claude.storage._read_from_keychain",
            return_value=None,
        ):
            loaded = await storage.load()

        assert loaded is None
