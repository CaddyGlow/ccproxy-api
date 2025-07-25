"""OS keyring storage implementation for token storage."""

import json

from structlog import get_logger

from ccproxy.auth.exceptions import (
    CredentialsStorageError,
)
from ccproxy.auth.models import ClaudeCredentials
from ccproxy.auth.storage.base import TokenStorage


logger = get_logger(__name__)


class KeyringTokenStorage(TokenStorage):
    """OS keyring storage implementation for Claude credentials."""

    def __init__(
        self, service_name: str = "claude-code-proxy", username: str = "default"
    ):
        """Initialize keyring storage.

        Args:
            service_name: Name of the service in the keyring
            username: Username to associate with the stored credentials
        """
        self.service_name = service_name
        self.username = username

    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from the OS keyring.

        Returns:
            Parsed credentials if found and valid, None otherwise

        Raises:
            CredentialsStorageError: If the stored data is invalid
            CredentialsStorageError: If there's an error reading from keyring
        """
        try:
            import keyring
        except ImportError as e:
            raise CredentialsStorageError(
                "keyring package is required for keyring storage. "
                "Install it with: pip install keyring"
            ) from e

        try:
            logger.debug(
                "credentials_load_start",
                source="keyring",
                service_name=self.service_name,
            )
            password = keyring.get_password(self.service_name, self.username)

            if password is None:
                logger.debug(
                    "credentials_not_found",
                    source="keyring",
                    service_name=self.service_name,
                )
                return None

            # Parse the stored JSON
            data = json.loads(password)
            credentials = ClaudeCredentials.model_validate(data)

            self._log_credential_details(credentials)
            return credentials

        except json.JSONDecodeError as e:
            raise CredentialsStorageError(
                f"Failed to parse credentials from keyring: {e}"
            ) from e
        except Exception as e:
            raise CredentialsStorageError(
                f"Error loading credentials from keyring: {e}"
            ) from e

    def _log_credential_details(self, credentials: ClaudeCredentials) -> None:
        """Log credential details safely."""
        oauth_token = credentials.claude_ai_oauth
        logger.debug(
            "credentials_load_completed",
            source="keyring",
            subscription_type=oauth_token.subscription_type,
            expires_at=str(oauth_token.expires_at_datetime),
            is_expired=oauth_token.is_expired,
            scopes=oauth_token.scopes,
        )

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to the OS keyring.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error writing to keyring
        """
        try:
            import keyring
        except ImportError as e:
            raise CredentialsStorageError(
                "keyring package is required for keyring storage. "
                "Install it with: pip install keyring"
            ) from e

        try:
            # Convert to JSON string
            data = credentials.model_dump(by_alias=True)
            json_data = json.dumps(data)

            # Store in keyring
            keyring.set_password(self.service_name, self.username, json_data)

            logger.debug(
                "credentials_save_completed",
                source="keyring",
                service_name=self.service_name,
            )
            return True

        except Exception as e:
            raise CredentialsStorageError(
                f"Error saving credentials to keyring: {e}"
            ) from e

    async def exists(self) -> bool:
        """Check if credentials exist in the keyring.

        Returns:
            True if credentials exist, False otherwise
        """
        try:
            import keyring
        except ImportError:
            return False

        try:
            password = keyring.get_password(self.service_name, self.username)
            return password is not None
        except Exception:
            return False

    async def delete(self) -> bool:
        """Delete credentials from the keyring.

        Returns:
            True if deleted successfully, False otherwise

        Raises:
            CredentialsStorageError: If there's an error deleting from keyring
        """
        try:
            import keyring
        except ImportError as e:
            raise CredentialsStorageError(
                "keyring package is required for keyring storage. "
                "Install it with: pip install keyring"
            ) from e

        try:
            if await self.exists():
                keyring.delete_password(self.service_name, self.username)
                logger.debug(
                    "credentials_delete_completed",
                    source="keyring",
                    service_name=self.service_name,
                )
                return True
            return False
        except Exception as e:
            raise CredentialsStorageError(
                f"Error deleting credentials from keyring: {e}"
            ) from e

    def get_location(self) -> str:
        """Get the storage location description.

        Returns:
            Description of the keyring storage location
        """
        return f"OS keyring (service: {self.service_name}, user: {self.username})"
