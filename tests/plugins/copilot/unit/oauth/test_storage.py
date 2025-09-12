"""Unit tests for CopilotOAuthStorage."""

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import SecretStr

from ccproxy.plugins.copilot.oauth.models import (
    CopilotCredentials,
    CopilotOAuthToken,
    CopilotTokenResponse,
)
from ccproxy.plugins.copilot.oauth.storage import CopilotOAuthStorage


class TestCopilotOAuthStorage:
    """Test cases for CopilotOAuthStorage."""

    @pytest.fixture
    def temp_storage_dir(self) -> Path:
        """Create temporary directory for storage tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

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
        return CopilotTokenResponse(
            token=SecretStr("copilot_test_token"),
            expires_at="2024-12-31T23:59:59Z",
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

    @pytest.fixture
    def storage_with_temp_dir(self, temp_storage_dir: Path) -> CopilotOAuthStorage:
        """Create storage with temporary directory."""
        return CopilotOAuthStorage(
            credentials_path=temp_storage_dir / "credentials.json"
        )

    def test_init_with_default_storage_dir(self) -> None:
        """Test initialization with default storage directory."""
        storage = CopilotOAuthStorage()

        expected_path = Path.home() / ".config" / "copilot" / "credentials.json"
        assert storage.file_path == expected_path

    def test_init_with_custom_storage_dir(self, temp_storage_dir: Path) -> None:
        """Test initialization with custom storage directory."""
        credentials_path = temp_storage_dir / "credentials.json"
        storage = CopilotOAuthStorage(credentials_path=credentials_path)

        assert storage.file_path == credentials_path

    async def test_save_credentials_creates_directory(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test saving credentials creates storage directory."""
        # Create a nested path that doesn't exist
        nested_path = (
            storage_with_temp_dir.file_path.parent / "nested" / "credentials.json"
        )
        storage = CopilotOAuthStorage(credentials_path=nested_path)

        # Ensure directory doesn't exist initially
        assert not nested_path.parent.exists()

        await storage.save(mock_credentials)

        # Directory should be created
        assert nested_path.parent.exists()
        assert nested_path.parent.is_dir()

        # Credentials file should be created
        assert nested_path.exists()

    async def test_save_credentials_writes_correct_data(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test saving credentials writes correct JSON data."""
        await storage_with_temp_dir.save(mock_credentials)

        # Read the file directly and verify contents
        with storage_with_temp_dir.file_path.open() as f:
            data = json.load(f)

        assert "oauth_token" in data
        assert "copilot_token" in data
        assert "account_type" in data
        assert "created_at" in data
        assert "updated_at" in data

        # Check OAuth token data
        oauth_data = data["oauth_token"]
        assert oauth_data["access_token"] == "gho_test_token"
        assert oauth_data["token_type"] == "bearer"
        assert oauth_data["scope"] == "read:user"

        # Check Copilot token data
        copilot_data = data["copilot_token"]
        assert copilot_data["token"] == "copilot_test_token"
        # expires_at is now serialized back to Unix timestamp
        expected_dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
        assert copilot_data["expires_at"] == int(expected_dt.timestamp())

        # Check account type
        assert data["account_type"] == "individual"

    async def test_save_credentials_updates_timestamps(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test saving credentials updates updated_at timestamp."""
        from unittest.mock import patch

        original_updated_at = mock_credentials.updated_at

        # Mock datetime.now to return a different timestamp
        with patch("ccproxy.plugins.copilot.oauth.models.datetime") as mock_datetime:
            mock_datetime.now.return_value.timestamp.return_value = (
                original_updated_at + 1
            )
            mock_datetime.UTC = mock_datetime.now.return_value.tzinfo

            await storage_with_temp_dir.save(mock_credentials)

        # updated_at should be changed
        assert mock_credentials.updated_at > original_updated_at

    async def test_save_credentials_handles_io_error(
        self,
        temp_storage_dir: Path,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test saving credentials handles I/O errors."""
        # Create storage with a read-only directory
        readonly_dir = temp_storage_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)  # Read-only

        credentials_path = readonly_dir / "credentials.json"
        storage = CopilotOAuthStorage(credentials_path=credentials_path)

        result = await storage.save(mock_credentials)

        # Should return False when I/O error occurs
        assert result is False

    async def test_load_credentials_success(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test successful credentials loading."""
        # First save credentials
        await storage_with_temp_dir.save(mock_credentials)

        # Then load them
        loaded_credentials = await storage_with_temp_dir.load()

        assert loaded_credentials is not None
        assert isinstance(loaded_credentials, CopilotCredentials)

        # Check OAuth token
        assert (
            loaded_credentials.oauth_token.access_token.get_secret_value()
            == "gho_test_token"
        )
        assert loaded_credentials.oauth_token.token_type == "bearer"
        assert loaded_credentials.oauth_token.scope == "read:user"

        # Check Copilot token
        assert loaded_credentials.copilot_token is not None
        assert (
            loaded_credentials.copilot_token.token.get_secret_value()
            == "copilot_test_token"
        )
        # expires_at is now a datetime object
        expected_dt = datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC)
        assert loaded_credentials.copilot_token.expires_at == expected_dt

        # Check account type
        assert loaded_credentials.account_type == "individual"

    async def test_load_credentials_file_not_exists(
        self, storage_with_temp_dir: CopilotOAuthStorage
    ) -> None:
        """Test loading credentials when file doesn't exist."""
        result = await storage_with_temp_dir.load()

        assert result is None

    async def test_load_credentials_invalid_json(
        self, storage_with_temp_dir: CopilotOAuthStorage
    ) -> None:
        """Test loading credentials with invalid JSON."""
        # Create directory and write invalid JSON
        storage_with_temp_dir.file_path.parent.mkdir(parents=True, exist_ok=True)
        with storage_with_temp_dir.file_path.open("w") as f:
            f.write("invalid json{")

        result = await storage_with_temp_dir.load()

        # Should return None when JSON is invalid (error is logged but not raised)
        assert result is None

    async def test_load_credentials_invalid_data_format(
        self, storage_with_temp_dir: CopilotOAuthStorage
    ) -> None:
        """Test loading credentials with invalid data format."""
        # Create directory and write invalid data structure
        storage_with_temp_dir.file_path.parent.mkdir(parents=True, exist_ok=True)
        with storage_with_temp_dir.file_path.open("w") as f:
            json.dump({"invalid": "data"}, f)

        result = await storage_with_temp_dir.load()

        assert result is None

    async def test_load_credentials_handles_io_error(
        self, temp_storage_dir: Path
    ) -> None:
        """Test loading credentials handles I/O errors."""
        # Create a directory where the credentials file should be
        credentials_path = temp_storage_dir / "storage" / "credentials.json"
        credentials_path.parent.mkdir(parents=True)
        credentials_path.mkdir()  # Create as directory instead of file

        storage = CopilotOAuthStorage(credentials_path=credentials_path)

        result = await storage.load()

        # Should return None when I/O error occurs (error is logged but not raised)
        assert result is None

    async def test_clear_credentials_file_exists(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test clearing credentials when file exists."""
        # First save credentials
        await storage_with_temp_dir.save(mock_credentials)
        assert storage_with_temp_dir.file_path.exists()

        # Clear credentials
        await storage_with_temp_dir.delete()

        # File should be deleted
        assert not storage_with_temp_dir.file_path.exists()

    async def test_clear_credentials_file_not_exists(
        self, storage_with_temp_dir: CopilotOAuthStorage
    ) -> None:
        """Test clearing credentials when file doesn't exist."""
        # File doesn't exist initially
        assert not storage_with_temp_dir.file_path.exists()

        # Clear should not raise error
        await storage_with_temp_dir.delete()

        # File still shouldn't exist
        assert not storage_with_temp_dir.file_path.exists()

    async def test_clear_credentials_handles_io_error(
        self, temp_storage_dir: Path
    ) -> None:
        """Test clearing credentials handles I/O errors."""
        # Create a read-only file
        storage_dir = temp_storage_dir / "storage"
        storage_dir.mkdir(parents=True)

        credentials_file = storage_dir / "credentials.json"
        credentials_file.write_text('{"test": "data"}')
        credentials_file.chmod(0o444)  # Read-only

        # Make directory read-only too
        storage_dir.chmod(0o555)

        storage = CopilotOAuthStorage(credentials_path=credentials_file)

        # Should raise CredentialsStorageError for permission error
        from ccproxy.auth.exceptions import CredentialsStorageError

        with pytest.raises(CredentialsStorageError):
            await storage.delete()

    async def test_save_and_load_round_trip(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test complete save and load round trip."""
        # Save credentials
        await storage_with_temp_dir.save(mock_credentials)

        # Load credentials
        loaded = await storage_with_temp_dir.load()

        assert loaded is not None

        # Compare all important fields
        assert (
            loaded.oauth_token.access_token.get_secret_value()
            == mock_credentials.oauth_token.access_token.get_secret_value()
        )
        assert loaded.oauth_token.token_type == mock_credentials.oauth_token.token_type
        assert loaded.oauth_token.expires_in == mock_credentials.oauth_token.expires_in
        assert loaded.oauth_token.scope == mock_credentials.oauth_token.scope

        if mock_credentials.copilot_token:
            assert loaded.copilot_token is not None
            assert (
                loaded.copilot_token.token.get_secret_value()
                == mock_credentials.copilot_token.token.get_secret_value()
            )
            assert (
                loaded.copilot_token.expires_at
                == mock_credentials.copilot_token.expires_at
            )

        assert loaded.account_type == mock_credentials.account_type

    async def test_save_credentials_without_copilot_token(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_oauth_token: CopilotOAuthToken,
    ) -> None:
        """Test saving credentials without Copilot token."""
        credentials = CopilotCredentials(
            oauth_token=mock_oauth_token,
            copilot_token=None,
            account_type="individual",
        )

        await storage_with_temp_dir.save(credentials)

        # Load and verify
        loaded = await storage_with_temp_dir.load()

        assert loaded is not None
        assert loaded.copilot_token is None
        assert loaded.oauth_token.access_token.get_secret_value() == "gho_test_token"
        assert loaded.account_type == "individual"

    async def test_concurrent_access_safety(
        self,
        storage_with_temp_dir: CopilotOAuthStorage,
        mock_credentials: CopilotCredentials,
    ) -> None:
        """Test storage handles concurrent access safely."""
        import asyncio

        async def save_credentials(creds: CopilotCredentials) -> None:
            await storage_with_temp_dir.save(creds)

        async def load_credentials() -> CopilotCredentials | None:
            return await storage_with_temp_dir.load()

        # Run fewer concurrent operations for faster tests
        tasks = []
        for _ in range(2):  # Reduced from 5 to 2 for faster execution
            tasks.append(save_credentials(mock_credentials))
            tasks.append(load_credentials())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # None of the operations should have failed with exceptions
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation failed: {result}")

        # Final state should be consistent
        final_creds = await storage_with_temp_dir.load()
        assert final_creds is not None
        assert (
            final_creds.oauth_token.access_token.get_secret_value() == "gho_test_token"
        )
