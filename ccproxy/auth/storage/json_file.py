"""JSON file storage implementation for token storage."""

import contextlib
import json
from pathlib import Path
from typing import Optional

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class JsonFileTokenStorage(TokenStorage):
    """JSON file storage implementation for Claude credentials."""

    def __init__(self, file_path: Path):
        """Initialize JSON file storage.

        Args:
            file_path: Path to the JSON credentials file
        """
        self.file_path = file_path

    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from JSON file.

        Returns:
            Parsed credentials if found and valid, None otherwise

        Raises:
            CredentialsInvalidError: If the JSON file is invalid
            CredentialsStorageError: If there's an error reading the file
        """
        if not await self.exists():
            logger.debug(f"Credentials file not found: {self.file_path}")
            return None

        try:
            logger.debug(f"Loading credentials from file: {self.file_path}")
            with self.file_path.open() as f:
                data = json.load(f)

            credentials = ClaudeCredentials.model_validate(data)
            self._log_credential_details(credentials)

            return credentials

        except json.JSONDecodeError as e:
            raise CredentialsInvalidError(
                f"Failed to parse credentials file {self.file_path}: {e}"
            ) from e
        except Exception as e:
            raise CredentialsStorageError(
                f"Error loading credentials from {self.file_path}: {e}"
            ) from e

    def _log_credential_details(self, credentials: ClaudeCredentials) -> None:
        """Log credential details safely."""
        oauth_token = credentials.claude_ai_oauth
        logger.debug("Successfully loaded credentials:")
        logger.debug(f"  - Subscription type: {oauth_token.subscription_type}")
        logger.debug(f"  - Token expires at: {oauth_token.expires_at_datetime}")
        logger.debug(f"  - Token expired: {oauth_token.is_expired}")
        logger.debug(f"  - Scopes: {oauth_token.scopes}")

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to JSON file.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error writing the file
        """
        try:
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict with proper aliases
            data = credentials.model_dump(by_alias=True)

            # Use atomic write: write to temp file then rename
            temp_path = self.file_path.with_suffix(".tmp")

            try:
                with temp_path.open("w") as f:
                    json.dump(data, f, indent=2)

                # Set appropriate file permissions (read/write for owner only)
                temp_path.chmod(0o600)

                # Atomically replace the original file
                Path.replace(temp_path, self.file_path)

                logger.debug(
                    f"Successfully saved credentials to file: {self.file_path}"
                )
                return True
            except Exception:
                # Clean up temp file if it exists
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()
                raise

        except Exception as e:
            raise CredentialsStorageError(f"Error saving credentials: {e}") from e

    async def exists(self) -> bool:
        """Check if credentials file exists.

        Returns:
            True if file exists, False otherwise
        """
        return self.file_path.exists() and self.file_path.is_file()

    async def delete(self) -> bool:
        """Delete credentials from file.

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error deleting the file
        """
        try:
            if await self.exists():
                self.file_path.unlink()
                logger.debug(f"Deleted credentials file: {self.file_path}")
                return True
            return False
        except Exception as e:
            raise CredentialsStorageError(f"Error deleting credentials: {e}") from e

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Path to the JSON file
        """
        return str(self.file_path)
