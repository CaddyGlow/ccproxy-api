"""JSON file storage backend for credentials."""

import json
from pathlib import Path
from typing import Any, Optional

import keyring

from ccproxy.services.credentials.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.services.credentials.models import ClaudeCredentials
from ccproxy.services.credentials.storage import CredentialsStorageBackend
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


class JsonFileStorage(CredentialsStorageBackend):
    """JSON file storage backend for Claude credentials with optional keyring support."""

    def __init__(self, file_path: Path):
        """Initialize JSON file storage.

        Args:
            file_path: Path to the JSON credentials file
        """
        self.file_path = file_path
        self.keyring_service = "ccproxy"
        self.keyring_username = "credentials"
        self._keyring_available = True

        if self._keyring_available:
            logger.debug("Keyring support is available")
        else:
            logger.debug("Keyring support is not available, using file storage only")

    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from keyring (if available) or JSON file.

        Returns:
            Parsed credentials if found and valid, None otherwise

        Raises:
            CredentialsInvalidError: If the JSON file is invalid
            CredentialsStorageError: If there's an error reading the file
        """
        # Try to load from keyring first if available
        if self._keyring_available:
            try:
                logger.debug("Attempting to load credentials from keyring")
                creds_json = keyring.get_password(
                    self.keyring_service, self.keyring_username
                )
                if creds_json:
                    logger.debug("Found credentials in keyring")
                    data = json.loads(creds_json)
                    credentials = ClaudeCredentials.model_validate(data)
                    self._log_credential_details(credentials)
                    return credentials
                else:
                    logger.debug("No credentials found in keyring")
            except Exception as e:
                logger.warning(f"Failed to load credentials from keyring: {e}")
                # Fall through to file loading

        # Load from file
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
        """Save credentials to keyring (if available) and JSON file.

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

            # Save to keyring if available
            if self._keyring_available:
                try:
                    logger.debug("Saving credentials to keyring")
                    creds_json = json.dumps(data)
                    keyring.set_password(
                        self.keyring_service, self.keyring_username, creds_json
                    )
                    logger.debug("Successfully saved credentials to keyring")
                except Exception as e:
                    logger.warning(f"Failed to save credentials to keyring: {e}")
                    # Continue to save to file

            # Always save to file as well (for compatibility and backup)
            with self.file_path.open("w") as f:
                json.dump(data, f, indent=2)

            # Set appropriate file permissions (read/write for owner only)
            self.file_path.chmod(0o600)

            logger.debug(f"Successfully saved credentials to file: {self.file_path}")
            return True

        except Exception as e:
            raise CredentialsStorageError(f"Error saving credentials: {e}") from e

    async def exists(self) -> bool:
        """Check if credentials file exists.

        Returns:
            True if file exists, False otherwise
        """
        return self.file_path.exists() and self.file_path.is_file()

    async def delete(self) -> bool:
        """Delete credentials from keyring (if available) and file.

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error deleting the file
        """
        try:
            deleted = False

            # Delete from keyring if available
            if self._keyring_available:
                try:
                    logger.debug("Deleting credentials from keyring")
                    keyring.delete_password(self.keyring_service, self.keyring_username)
                    logger.debug("Deleted credentials from keyring")
                    deleted = True
                except Exception as e:
                    logger.debug(f"No credentials in keyring to delete or error: {e}")

            # Delete file
            if await self.exists():
                self.file_path.unlink()
                logger.debug(f"Deleted credentials file: {self.file_path}")
                deleted = True

            return deleted
        except Exception as e:
            raise CredentialsStorageError(f"Error deleting credentials: {e}") from e

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Path to the JSON file and keyring status
        """
        if self._keyring_available:
            return f"{self.file_path} (with keyring support)"
        return str(self.file_path)
