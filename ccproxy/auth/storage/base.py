"""Abstract base class for token storage."""

import asyncio
import contextlib
import json
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

from ccproxy.auth.exceptions import CredentialsInvalidError, CredentialsStorageError
from ccproxy.auth.models.credentials import BaseCredentials
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)

CredentialsT = TypeVar("CredentialsT", bound=BaseCredentials)


class TokenStorage(ABC, Generic[CredentialsT]):
    """Abstract interface for token storage operations.

    This is a generic interface that can work with any credential type
    that extends BaseModel (e.g., ClaudeCredentials, OpenAICredentials).
    """

    @abstractmethod
    async def load(self) -> CredentialsT | None:
        """Load credentials from storage.

        Returns:
            Parsed credentials if found and valid, None otherwise
        """
        pass

    @abstractmethod
    async def save(self, credentials: CredentialsT) -> bool:
        """Save credentials to storage.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        pass

    @abstractmethod
    async def exists(self) -> bool:
        """Check if credentials exist in storage.

        Returns:
            True if credentials exist, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self) -> bool:
        """Delete credentials from storage.

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Human-readable description of where credentials are stored
        """
        pass


class BaseJsonStorage(TokenStorage[CredentialsT], Generic[CredentialsT]):
    """Base class for JSON file storage implementations.

    This class provides common JSON read/write operations with error handling,
    atomic writes, and proper permission management.

    This is a generic class that can work with any credential type.
    """

    def __init__(self, file_path: Path, enable_backups: bool = True):
        """Initialize JSON storage.

        Args:
            file_path: Path to JSON file for storage
            enable_backups: Whether to create backups before overwriting
        """
        self.file_path = file_path
        self.enable_backups = enable_backups

    async def _read_json(self) -> dict[str, Any]:
        """Read JSON data from file with error handling.

        Returns:
            Parsed JSON data or empty dict if file doesn't exist

        Raises:
            CredentialsInvalidError: If JSON is invalid
            CredentialsStorageError: If file cannot be read
        """
        if not await self.exists():
            return {}

        try:
            # Run file I/O in thread pool to avoid blocking
            def read_file() -> dict[str, Any]:
                with self.file_path.open("r") as f:
                    return json.load(f)  # type: ignore[no-any-return]

            data = await asyncio.to_thread(read_file)
            return data

        except json.JSONDecodeError as e:
            logger.error(
                "json_decode_error",
                path=str(self.file_path),
                error=str(e),
                line=e.lineno,
                exc_info=e,
            )
            raise CredentialsInvalidError(
                f"Invalid JSON in {self.file_path}: {e}"
            ) from e

        except FileNotFoundError:
            # File was deleted between exists() check and read
            return {}

        except PermissionError as e:
            logger.error(
                "permission_denied",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Permission denied: {self.file_path}") from e

        except OSError as e:
            logger.error(
                "file_read_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Error reading {self.file_path}: {e}") from e

    async def _create_backup(self) -> bool:
        """Create a timestamped backup of the current file.

        Returns:
            True if backup was created successfully, False otherwise
        """
        if not await self.exists():
            return False

        try:
            # Generate backup filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_name = f"{self.file_path.name}.{timestamp}.bak"
            backup_path = self.file_path.parent / backup_name

            # Copy file to backup location
            await asyncio.to_thread(shutil.copy2, self.file_path, backup_path)

            logger.info(
                "backup_created",
                original=str(self.file_path),
                backup=str(backup_path),
                category="auth",
            )
            return True

        except Exception as e:
            logger.warning(
                "backup_failed",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return False

    async def _write_json(self, data: dict[str, Any]) -> None:
        """Write JSON data to file atomically with error handling.

        This method performs atomic writes by writing to a temporary file
        first, then renaming it to the target file. If backups are enabled
        and the file exists, a backup will be created before overwriting.

        Args:
            data: Data to write as JSON

        Raises:
            CredentialsStorageError: If file cannot be written
        """
        # Create backup if enabled and file exists
        if self.enable_backups and await self.exists():
            await self._create_backup()

        temp_path = self.file_path.with_suffix(".tmp")

        try:
            # Ensure parent directory exists
            await asyncio.to_thread(
                self.file_path.parent.mkdir,
                parents=True,
                exist_ok=True,
            )

            # Run file I/O in thread pool to avoid blocking
            def write_file() -> None:
                # Write to temporary file
                with temp_path.open("w") as f:
                    json.dump(data, f, indent=2)

                # Set restrictive permissions (read/write for owner only)
                temp_path.chmod(0o600)

                # Atomic rename
                temp_path.replace(self.file_path)

            await asyncio.to_thread(write_file)

            logger.debug(
                "json_write_success",
                path=str(self.file_path),
                size=len(json.dumps(data)),
            )

        except (TypeError, ValueError) as e:
            logger.error(
                "json_encode_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Failed to encode JSON: {e}") from e

        except PermissionError as e:
            logger.error(
                "permission_denied",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Permission denied: {self.file_path}") from e

        except OSError as e:
            logger.error(
                "file_write_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Error writing {self.file_path}: {e}") from e

        finally:
            # Clean up temp file if it exists
            if temp_path.exists():
                with contextlib.suppress(Exception):
                    temp_path.unlink()

    async def exists(self) -> bool:
        """Check if credentials file exists.

        Returns:
            True if file exists, False otherwise
        """
        # Run file system check in thread pool for consistency
        file_exists = await asyncio.to_thread(
            lambda: self.file_path.exists() and self.file_path.is_file()
        )

        logger.debug(
            "auth_file_existence_check",
            file_path=str(self.file_path),
            exists=file_exists,
            category="auth",
        )

        return file_exists

    async def delete(self) -> bool:
        """Delete credentials file.

        Returns:
            True if deleted successfully, False if file didn't exist

        Raises:
            CredentialsStorageError: If file cannot be deleted
        """
        try:
            if await self.exists():
                await asyncio.to_thread(self.file_path.unlink)
                logger.debug("file_deleted", path=str(self.file_path))
                return True
            return False

        except PermissionError as e:
            logger.error(
                "permission_denied",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(f"Permission denied: {self.file_path}") from e

        except OSError as e:
            logger.error(
                "file_delete_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(
                f"Error deleting {self.file_path}: {e}"
            ) from e

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Path to the JSON file
        """
        return str(self.file_path)
