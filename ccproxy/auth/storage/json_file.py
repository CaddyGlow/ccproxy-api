"""JSON file storage implementation for token storage."""

import contextlib
import json
from pathlib import Path
from typing import Optional

from structlog import get_logger

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.storage.base import TokenStorage


logger = get_logger(__name__)


class JsonFileTokenStorage(TokenStorage):
    """JSON file storage implementation for Claude credentials with keyring fallback."""

    def __init__(self, file_path: Path):
        """Initialize JSON file storage.

        Args:
            file_path: Path to the JSON credentials file
        """
        self.file_path = file_path

    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from JSON file .

        Returns:
            Parsed credentials if found and valid, None otherwise

        Raises:
            CredentialsInvalidError: If the JSON file is invalid
            CredentialsStorageError: If there's an error reading the file
        """
        if not await self.exists():
            logger.debug("credentials_file_not_found", path=str(self.file_path))
            return None

        try:
            logger.debug(
                "credentials_load_start", source="file", path=str(self.file_path)
            )
            with self.file_path.open() as f:
                data = json.load(f)

            credentials = ClaudeCredentials.model_validate(data)
            logger.debug("credentials_load_completed", source="file")

            return credentials

        except json.JSONDecodeError as e:
            raise CredentialsInvalidError(
                f"Failed to parse credentials file {self.file_path}: {e}"
            ) from e
        except Exception as e:
            raise CredentialsStorageError(
                f"Error loading credentials from {self.file_path}: {e}"
            ) from e

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to both keyring and JSON file.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error writing the file
        """
        try:
            # Convert to dict with proper aliases
            data = credentials.model_dump(by_alias=True, mode="json")

            # Always save to file as well
            # Ensure parent directory exists
            self.file_path.parent.mkdir(parents=True, exist_ok=True)

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
                    "credentials_save_completed",
                    source="file",
                    path=str(self.file_path),
                )
                return True
            except Exception as e:
                raise
            finally:
                # Clean up temp file if it exists
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()

        except Exception as e:
            raise CredentialsStorageError(f"Error saving credentials: {e}") from e

    async def exists(self) -> bool:
        """Check if credentials file exists.

        Returns:
            True if file exists, False otherwise
        """
        return self.file_path.exists() and self.file_path.is_file()

    async def delete(self) -> bool:
        """Delete credentials from both keyring and file.

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error deleting the file
        """
        deleted = False

        # Delete from file
        try:
            if await self.exists():
                self.file_path.unlink()
                logger.debug(
                    "credentials_delete_completed",
                    source="file",
                    path=str(self.file_path),
                )
                deleted = True
        except Exception as e:
            if not deleted:  # Only raise if we failed to delete from both
                raise CredentialsStorageError(f"Error deleting credentials: {e}") from e
            logger.debug("credentials_delete_partial", source="file", error=str(e))

        return deleted

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Path to the JSON file with keyring info if available
        """
        return str(self.file_path)
