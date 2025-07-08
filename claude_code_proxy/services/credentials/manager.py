"""Credentials manager for coordinating storage and OAuth operations."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from claude_code_proxy.services.credentials.config import CredentialsConfig
from claude_code_proxy.services.credentials.exceptions import (
    CredentialsExpiredError,
    CredentialsNotFoundError,
)
from claude_code_proxy.services.credentials.json_storage import JsonFileStorage
from claude_code_proxy.services.credentials.models import ClaudeCredentials, UserProfile
from claude_code_proxy.services.credentials.oauth_client import OAuthClient
from claude_code_proxy.services.credentials.storage import CredentialsStorageBackend
from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)


class CredentialsManager:
    """Manager for Claude credentials with storage and OAuth support."""

    def __init__(
        self,
        config: CredentialsConfig | None = None,
        storage: CredentialsStorageBackend | None = None,
        oauth_client: OAuthClient | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """Initialize credentials manager.

        Args:
            config: Credentials configuration (uses defaults if not provided)
            storage: Storage backend (uses JSON file storage if not provided)
            oauth_client: OAuth client (creates one if not provided)
            http_client: HTTP client for OAuth operations
        """
        self.config = config or CredentialsConfig()
        self._storage = storage
        self._oauth_client = oauth_client
        self._http_client = http_client
        self._owns_http_client = http_client is None

        # Initialize OAuth client if not provided
        if self._oauth_client is None:
            self._oauth_client = OAuthClient(
                config=self.config.oauth,
                http_client=self._http_client,
            )

    async def __aenter__(self) -> "CredentialsManager":
        """Async context manager entry."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
            if self._oauth_client:
                self._oauth_client._http_client = self._http_client
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owns_http_client and self._http_client:
            await self._http_client.aclose()

    @property
    def storage(self) -> CredentialsStorageBackend:
        """Get the storage backend, creating default if needed."""
        if self._storage is None:
            # Find first existing credentials file or use first path
            for path_str in self.config.storage_paths:
                path = Path(path_str).expanduser()
                if path.exists():
                    self._storage = JsonFileStorage(path)
                    break
            else:
                # Use first path as default
                self._storage = JsonFileStorage(
                    Path(self.config.storage_paths[0]).expanduser()
                )
        return self._storage

    async def find_credentials_file(self) -> Path | None:
        """Find existing credentials file in configured paths.

        Returns:
            Path to credentials file if found, None otherwise
        """
        for path_str in self.config.storage_paths:
            path = Path(path_str).expanduser()
            logger.debug(f"Checking: {path}")
            if path.exists() and path.is_file():
                logger.info(f"Found credentials file at: {path}")
                return path
            else:
                logger.debug(f"Not found: {path}")

        logger.warning("No credentials file found in any searched locations:")
        for path_str in self.config.storage_paths:
            logger.warning(f"  - {path_str}")
        return None

    async def load(self) -> ClaudeCredentials | None:
        """Load credentials from storage.

        Returns:
            Credentials if found and valid, None otherwise
        """
        try:
            return await self.storage.load()
        except Exception as e:
            logger.error(f"Error loading credentials: {e}")
            return None

    async def save(self, credentials: ClaudeCredentials) -> bool:
        """Save credentials to storage.

        Args:
            credentials: Credentials to save

        Returns:
            True if saved successfully, False otherwise
        """
        try:
            return await self.storage.save(credentials)
        except Exception as e:
            logger.error(f"Error saving credentials: {e}")
            return False

    async def login(self) -> ClaudeCredentials:
        """Perform OAuth login and save credentials.

        Returns:
            New credentials from login

        Raises:
            OAuthLoginError: If login fails
        """
        if self._oauth_client is None:
            raise RuntimeError("OAuth client not initialized")
        credentials = await self._oauth_client.login()
        await self.save(credentials)
        return credentials

    async def get_valid_credentials(self) -> ClaudeCredentials:
        """Get valid credentials, refreshing if necessary.

        Returns:
            Valid credentials

        Raises:
            CredentialsNotFoundError: If no credentials found
            CredentialsExpiredError: If credentials expired and refresh fails
        """
        credentials = await self.load()
        if not credentials:
            raise CredentialsNotFoundError("No credentials found. Please login first.")

        # Check if token needs refresh
        oauth_token = credentials.claude_ai_oauth

        # Calculate if we should refresh based on buffer
        if self.config.auto_refresh:
            buffer = timedelta(seconds=self.config.refresh_buffer_seconds)
            should_refresh = (
                datetime.now(UTC) + buffer >= oauth_token.expires_at_datetime
            )
        else:
            should_refresh = oauth_token.is_expired

        if should_refresh:
            logger.info("Token expired or expiring soon, refreshing...")
            try:
                if self._oauth_client is None:
                    raise RuntimeError("OAuth client not initialized")
                new_token = await self._oauth_client.refresh_token(
                    oauth_token.refresh_token
                )

                # Update credentials with new token
                new_token.subscription_type = oauth_token.subscription_type
                credentials.claude_ai_oauth = new_token

                # Save updated credentials
                await self.save(credentials)

                logger.info("Successfully refreshed token")
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}")
                if oauth_token.is_expired:
                    raise CredentialsExpiredError(
                        "Token expired and refresh failed. Please login again."
                    ) from e
                # If not expired yet but refresh failed, return existing token
                logger.warning("Using existing token despite failed refresh")

        return credentials

    async def get_access_token(self) -> str:
        """Get valid access token, refreshing if necessary.

        Returns:
            Access token string

        Raises:
            CredentialsNotFoundError: If no credentials found
            CredentialsExpiredError: If credentials expired and refresh fails
        """
        credentials = await self.get_valid_credentials()
        return credentials.claude_ai_oauth.access_token

    async def fetch_user_profile(self) -> UserProfile | None:
        """Fetch user profile information.

        Returns:
            UserProfile if successful, None otherwise
        """
        try:
            credentials = await self.get_valid_credentials()
            if self._oauth_client is None:
                raise RuntimeError("OAuth client not initialized")
            profile = await self._oauth_client.fetch_user_profile(
                credentials.claude_ai_oauth.access_token,
                credentials.claude_ai_oauth.refresh_token,
            )
            return profile
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return None

    async def validate(self) -> dict[str, str | bool | list[str] | None]:
        """Validate current credentials.

        Returns:
            Dictionary with validation results including:
            - valid: Whether credentials are found and valid
            - expired: Whether the token is expired
            - subscription_type: The subscription type if available
            - expires_at: Token expiration datetime
            - error: Error message if validation failed
        """
        try:
            credentials = await self.load()
            if not credentials:
                return {
                    "valid": False,
                    "error": f"No credentials file found in {', '.join(self.config.storage_paths)}",
                }

            oauth_token = credentials.claude_ai_oauth

            return {
                "valid": True,
                "expired": oauth_token.is_expired,
                "subscription_type": oauth_token.subscription_type,
                "expires_at": oauth_token.expires_at_datetime.isoformat(),
                "scopes": oauth_token.scopes,
            }

        except Exception as e:
            logger.exception("Error validating credentials")
            return {
                "valid": False,
                "error": str(e),
            }

    async def logout(self) -> bool:
        """Delete stored credentials.

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            return await self.storage.delete()
        except Exception as e:
            logger.error(f"Error deleting credentials: {e}")
            return False
