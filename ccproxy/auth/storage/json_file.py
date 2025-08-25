"""JSON file storage implementation for token storage."""

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ccproxy.auth.exceptions import (
    CredentialsInvalidError,
    CredentialsStorageError,
)
from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class JsonFileTokenStorage(TokenStorage[ClaudeCredentials]):
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
        # Get logger with request context at the start of the function
        logger = get_logger(__name__)

        if not await self.exists():
            logger.debug("credentials_file_not_found", path=str(self.file_path))
            return None

        try:
            logger.debug(
                "credentials_load_start", source="file", path=str(self.file_path)
            )

            # Run file I/O in thread pool to avoid blocking
            def read_file() -> dict[str, Any]:
                with self.file_path.open() as f:
                    return json.load(f)  # type: ignore[no-any-return]

            data = await asyncio.to_thread(read_file)
            credentials = ClaudeCredentials.model_validate(data)
            logger.debug("credentials_load_completed", source="file")

            return credentials

        except json.JSONDecodeError as e:
            raise CredentialsInvalidError(
                f"Failed to parse credentials file {self.file_path}: {e}"
            ) from e
        except FileNotFoundError as e:
            logger.error(
                "file_not_found", path=str(self.file_path), error=str(e), exc_info=e
            )
            raise CredentialsStorageError(
                f"Credentials file not found: {self.file_path}"
            ) from e
        except PermissionError as e:
            logger.error(
                "permission_denied", path=str(self.file_path), error=str(e), exc_info=e
            )
            raise CredentialsStorageError(
                f"Permission denied accessing credentials file: {self.file_path}"
            ) from e
        except UnicodeDecodeError as e:
            logger.error(
                "unicode_decode_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsInvalidError(
                f"Invalid file encoding in credentials file {self.file_path}: {e}"
            ) from e
        except OSError as e:
            logger.error(
                "file_io_error", path=str(self.file_path), error=str(e), exc_info=e
            )
            raise CredentialsStorageError(
                f"Error accessing credentials file {self.file_path}: {e}"
            ) from e
        except ValidationError as e:
            logger.error(
                "validation_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsInvalidError(
                f"Invalid credentials format in {self.file_path}: {e}"
            ) from e
        except Exception as e:
            logger.error(
                "unexpected_load_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(
                f"Unexpected error loading credentials from {self.file_path}: {e}"
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

            # Ensure parent directory exists - run in thread pool
            await asyncio.to_thread(
                self.file_path.parent.mkdir, parents=True, exist_ok=True
            )

            # Use atomic write: write to temp file then rename
            temp_path = self.file_path.with_suffix(".tmp")

            try:
                # Run file I/O operations in thread pool to avoid blocking
                def write_file() -> None:
                    with temp_path.open("w") as f:
                        json.dump(data, f, indent=2)

                    # Set appropriate file permissions (read/write for owner only)
                    temp_path.chmod(0o600)

                    # Atomically replace the original file
                    Path.replace(temp_path, self.file_path)

                await asyncio.to_thread(write_file)

                logger.debug(
                    "credentials_save_completed",
                    source="file",
                    path=str(self.file_path),
                )
                return True
            except FileNotFoundError as e:
                logger.error(
                    "temp_file_not_found", path=str(temp_path), error=str(e), exc_info=e
                )
                raise CredentialsStorageError(
                    f"Temporary file not found during save: {temp_path}"
                ) from e
            except PermissionError as e:
                logger.error(
                    "permission_denied", path=str(temp_path), error=str(e), exc_info=e
                )
                raise CredentialsStorageError(
                    f"Permission denied writing credentials file: {temp_path}"
                ) from e
            except UnicodeEncodeError as e:
                logger.error(
                    "unicode_encode_error",
                    path=str(temp_path),
                    error=str(e),
                    exc_info=e,
                )
                raise CredentialsStorageError(
                    f"Error encoding credentials data: {e}"
                ) from e
            except OSError as e:
                logger.error(
                    "file_io_error", path=str(temp_path), error=str(e), exc_info=e
                )
                raise CredentialsStorageError(
                    f"Error writing credentials file: {e}"
                ) from e
            except (TypeError, ValueError) as e:
                logger.error(
                    "json_encode_error",
                    path=str(temp_path),
                    error=str(e),
                    exc_info=e,
                )
                raise CredentialsStorageError(
                    f"Failed to encode credentials as JSON: {e}"
                ) from e
            except ValidationError as e:
                logger.error(
                    "validation_error",
                    path=str(temp_path),
                    error=str(e),
                    exc_info=e,
                )
                raise CredentialsInvalidError(f"Invalid credentials format: {e}") from e
            except Exception as e:
                logger.error(
                    "unexpected_save_error",
                    path=str(temp_path),
                    error=str(e),
                    exc_info=e,
                )
                raise
            finally:
                # Clean up temp file if it exists
                if temp_path.exists():
                    with contextlib.suppress(Exception):
                        temp_path.unlink()

        except CredentialsStorageError:
            raise  # Re-raise already handled storage errors
        except ValidationError as e:
            logger.error(
                "outer_validation_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsInvalidError(f"Invalid credentials format: {e}") from e
        except Exception as e:
            logger.error(
                "unexpected_outer_save_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            raise CredentialsStorageError(
                f"Unexpected error saving credentials: {e}"
            ) from e

    async def exists(self) -> bool:
        """Check if credentials file exists.

        Returns:
            True if file exists, False otherwise
        """
        # File system operations are typically fast enough not to warrant async handling,
        # but for consistency, we can run in thread pool if needed
        return await asyncio.to_thread(
            lambda: self.file_path.exists() and self.file_path.is_file()
        )

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
                # Run file deletion in thread pool to avoid blocking
                await asyncio.to_thread(self.file_path.unlink)
                logger.debug(
                    "credentials_delete_completed",
                    source="file",
                    path=str(self.file_path),
                )
                deleted = True
        except FileNotFoundError as e:
            logger.debug(
                "file_not_found_during_delete",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            # File not found is success for delete operation
        except PermissionError as e:
            logger.error(
                "permission_denied", path=str(self.file_path), error=str(e), exc_info=e
            )
            if not deleted:
                raise CredentialsStorageError(
                    f"Permission denied deleting credentials file: {self.file_path}"
                ) from e
            logger.debug("credentials_delete_partial", source="file", error=str(e))
        except OSError as e:
            logger.error(
                "file_io_error", path=str(self.file_path), error=str(e), exc_info=e
            )
            if not deleted:
                raise CredentialsStorageError(
                    f"Error deleting credentials file: {self.file_path}"
                ) from e
            logger.debug("credentials_delete_partial", source="file", error=str(e))
        except Exception as e:
            logger.error(
                "unexpected_delete_error",
                path=str(self.file_path),
                error=str(e),
                exc_info=e,
            )
            if not deleted:
                raise CredentialsStorageError(
                    f"Unexpected error deleting credentials: {e}"
                ) from e
            logger.debug("credentials_delete_partial", source="file", error=str(e))

        return deleted

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Path to the JSON file with keyring info if available
        """
        return str(self.file_path)
