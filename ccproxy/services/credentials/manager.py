"""Credentials manager for coordinating storage and OAuth operations."""

import asyncio
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from ccproxy.auth.exceptions import (
    CredentialsExpiredError,
    CredentialsNotFoundError,
)
from ccproxy.auth.models import (
    ClaudeCredentials,
    OAuthToken,
    UserProfile,
    ValidationResult,
)
from ccproxy.auth.storage import JsonFileTokenStorage as JsonFileStorage
from ccproxy.auth.storage import TokenStorage as CredentialsStorageBackend
from ccproxy.core.logging import get_logger
from ccproxy.services.credentials.config import CredentialsConfig
from ccproxy.services.credentials.oauth_client import OAuthClient


logger = get_logger(__name__)


class CredentialsManager:
    """Manager for Claude credentials with storage and OAuth support."""

    # ==================== Initialization ====================

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
        self._refresh_lock = asyncio.Lock()

        # Initialize OAuth client if not provided
        if self._oauth_client is None:
            self._oauth_client = OAuthClient(
                config=self.config.oauth,
            )

    async def __aenter__(self) -> "CredentialsManager":
        """Async context manager entry."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if self._owns_http_client and self._http_client:
            await self._http_client.aclose()

    # ==================== Storage Operations ====================

    @property
    def storage(self) -> CredentialsStorageBackend:
        """Get the storage backend, creating default if needed."""
        if self._storage is None:
            # Find first existing credentials file or use first path
            existing_path = self._find_existing_path()
            if existing_path:
                self._storage = JsonFileStorage(existing_path)
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

    # ==================== OAuth Operations ====================

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

        # Fetch and save user profile after successful login
        try:
            profile = await self._oauth_client.fetch_user_profile(
                credentials.claude_ai_oauth.access_token
            )
            if profile:
                # Save profile data
                await self._save_account_profile(profile)

                # Update subscription type based on profile
                determined_subscription = self._determine_subscription_type(profile)
                credentials.claude_ai_oauth.subscription_type = determined_subscription

                logger.debug(f"Set subscription type to: {determined_subscription}")
            else:
                logger.debug("No profile data available during login")
        except Exception as e:
            logger.warning(f"Failed to fetch profile during login: {e}")
            # Continue with login even if profile fetch fails

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
        should_refresh = self._should_refresh_token(oauth_token)

        if should_refresh:
            async with self._refresh_lock:
                # Re-check if refresh is still needed after acquiring lock
                # Another request might have already refreshed the token
                credentials = await self.load()
                if not credentials:
                    raise CredentialsNotFoundError(
                        "No credentials found. Please login first."
                    )

                oauth_token = credentials.claude_ai_oauth
                should_refresh = self._should_refresh_token(oauth_token)

                if should_refresh:
                    logger.info("Token expired or expiring soon, refreshing...")
                    try:
                        credentials = await self._refresh_token_with_profile(
                            credentials
                        )
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

    async def refresh_token(self) -> ClaudeCredentials:
        """Refresh the access token without checking expiration.

        This method directly refreshes the token regardless of whether it's expired.
        Useful for force-refreshing tokens or testing.

        Returns:
            Updated credentials with new token

        Raises:
            CredentialsNotFoundError: If no credentials found
            RuntimeError: If OAuth client not initialized
            ValueError: If no refresh token available
            Exception: If token refresh fails
        """
        credentials = await self.load()
        if not credentials:
            raise CredentialsNotFoundError("No credentials found. Please login first.")

        logger.info("Refreshing token (forced)")
        return await self._refresh_token_with_profile(credentials)

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
            )
            return profile
        except Exception as e:
            logger.error(
                f"Error fetching user profile: {e}",
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            return None

    async def get_account_profile(self) -> UserProfile | None:
        """Get saved account profile information.

        Returns:
            UserProfile if available, None otherwise
        """
        return await self._load_account_profile()

    # ==================== Validation and Management ====================

    async def validate(self) -> ValidationResult:
        """Validate current credentials.

        Returns:
            ValidationResult with credentials status and details
        """
        try:
            credentials = await self.load()
            if not credentials:
                return ValidationResult(
                    valid=False, expired=None, credentials=None, path=None
                )

            return ValidationResult(
                valid=True,
                expired=credentials.claude_ai_oauth.is_expired,
                credentials=credentials,
                path=self.storage.get_location(),
            )

        except Exception as e:
            logger.exception("Error validating credentials")
            return ValidationResult(
                valid=False, expired=None, credentials=None, path=None
            )

    async def logout(self) -> bool:
        """Delete stored credentials.

        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            # Delete both credentials and account profile
            success = await self.storage.delete()
            await self._delete_account_profile()
            return success
        except Exception as e:
            logger.error(f"Error deleting credentials: {e}")
            return False

    # ==================== Private Helper Methods ====================

    async def _get_account_profile_path(self) -> Path:
        """Get the path for account profile storage.

        Returns:
            Path to account.json file alongside credentials
        """
        # Use the same directory as credentials file but with account.json name
        credentials_path = self._find_existing_path()
        if credentials_path is None:
            # Use first path as default
            credentials_path = Path(self.config.storage_paths[0]).expanduser()

        # Replace filename with account.json
        return credentials_path.parent / "account.json"

    async def _save_account_profile(self, profile: UserProfile) -> bool:
        """Save account profile to account.json.

        Args:
            profile: User profile to save

        Returns:
            True if saved successfully
        """
        try:
            account_path = await self._get_account_profile_path()
            account_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert to dict and save as JSON
            profile_data = profile.model_dump()

            with account_path.open("w", encoding="utf-8") as f:
                json.dump(profile_data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved account profile to: {account_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving account profile: {e}")
            return False

    async def _load_account_profile(self) -> UserProfile | None:
        """Load account profile from account.json.

        Returns:
            User profile if found, None otherwise
        """
        try:
            account_path = await self._get_account_profile_path()

            if not account_path.exists():
                logger.debug("No account profile file found")
                return None

            with account_path.open("r", encoding="utf-8") as f:
                profile_data = json.load(f)

            return UserProfile.model_validate(profile_data)

        except Exception as e:
            logger.debug(f"Error loading account profile: {e}")
            return None

    async def _delete_account_profile(self) -> bool:
        """Delete account profile file.

        Returns:
            True if deleted successfully
        """
        try:
            account_path = await self._get_account_profile_path()
            if account_path.exists():
                account_path.unlink()
                logger.debug(f"Deleted account profile: {account_path}")
            return True
        except Exception as e:
            logger.debug(f"Error deleting account profile: {e}")
            return False

    def _determine_subscription_type(self, profile: UserProfile) -> str:
        """Determine subscription type from profile information.

        Args:
            profile: User profile with account information

        Returns:
            Subscription type string
        """
        if not profile.account:
            return "unknown"

        # Check account flags first
        if profile.account.has_claude_max:
            return "max"
        elif profile.account.has_claude_pro:
            return "pro"

        # Fallback to organization type
        if profile.organization and profile.organization.organization_type:
            org_type = profile.organization.organization_type.lower()
            if "max" in org_type:
                return "max"
            elif "pro" in org_type:
                return "pro"

        return "free"

    def _find_existing_path(self) -> Path | None:
        """Find first existing path from configured storage paths.

        Returns:
            Path if found, None otherwise
        """
        for path_str in self.config.storage_paths:
            path = Path(path_str).expanduser()
            if path.exists():
                return path
        return None

    def _should_refresh_token(self, oauth_token: OAuthToken) -> bool:
        """Check if token should be refreshed based on configuration.

        Args:
            oauth_token: Token to check

        Returns:
            True if token should be refreshed
        """
        if self.config.auto_refresh:
            buffer = timedelta(seconds=self.config.refresh_buffer_seconds)
            return datetime.now(UTC) + buffer >= oauth_token.expires_at_datetime
        else:
            return oauth_token.is_expired

    async def _refresh_token_with_profile(
        self, credentials: ClaudeCredentials
    ) -> ClaudeCredentials:
        """Refresh token and update profile information.

        Args:
            credentials: Current credentials with token to refresh

        Returns:
            Updated credentials with new token and profile info

        Raises:
            RuntimeError: If OAuth client not initialized
            ValueError: If no refresh token available
            Exception: If token refresh fails
        """
        if self._oauth_client is None:
            raise RuntimeError("OAuth client not initialized")

        oauth_token = credentials.claude_ai_oauth

        # Refresh the token
        token_response = await self._oauth_client.refresh_access_token(
            oauth_token.refresh_token
        )

        # Calculate expires_at from expires_in if provided
        expires_at = oauth_token.expires_at  # Start with existing value
        if token_response.expires_in:
            expires_at = int(
                (datetime.now(UTC).timestamp() + token_response.expires_in) * 1000
            )

        # Parse scopes from server response
        new_scopes = oauth_token.scopes  # Start with existing scopes
        if token_response.scope:
            new_scopes = token_response.scope.split()

        # Create new token preserving all server fields when available
        # Ensure we have valid refresh token
        if not token_response.refresh_token and not oauth_token.refresh_token:
            raise ValueError("No refresh token available")

        # Convert OAuthTokenResponse to OAuthToken format
        new_token = OAuthToken(
            accessToken=token_response.access_token,
            refreshToken=token_response.refresh_token or oauth_token.refresh_token,
            expiresAt=expires_at,
            scopes=new_scopes,
            subscriptionType=token_response.subscription_type
            or oauth_token.subscription_type,
            tokenType=token_response.token_type or oauth_token.token_type,
        )

        # Update credentials with new token
        credentials.claude_ai_oauth = new_token

        # Fetch user profile to update subscription type
        try:
            profile = await self._oauth_client.fetch_user_profile(
                new_token.access_token
            )
            if profile:
                # Save profile data
                await self._save_account_profile(profile)

                # Update subscription type based on profile
                determined_subscription = self._determine_subscription_type(profile)
                new_token.subscription_type = determined_subscription
                credentials.claude_ai_oauth = new_token

                logger.debug(f"Updated subscription type to: {determined_subscription}")
            else:
                logger.debug(
                    "No profile data available, keeping existing subscription type"
                )
        except Exception as e:
            logger.warning(f"Failed to fetch profile during token refresh: {e}")
            # Continue with token refresh even if profile fetch fails

        # Save updated credentials
        await self.save(credentials)

        logger.info("Successfully refreshed token")
        return credentials
