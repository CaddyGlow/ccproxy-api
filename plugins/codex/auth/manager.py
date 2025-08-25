"""OpenAI/Codex token manager implementation for the Codex plugin."""

from datetime import datetime
from pathlib import Path
from typing import Any

from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.models import (
    AccountInfo,
    OpenAICredentials,
    UserProfile,
)
from ccproxy.auth.oauth.registry import get_oauth_registry
from ccproxy.auth.storage.generic import GenericJsonStorage
from ccproxy.core.logging import get_plugin_logger

from .models import OpenAIProfileInfo, OpenAITokenWrapper


logger = get_plugin_logger()


class CodexTokenManager(BaseTokenManager[OpenAICredentials]):
    """Manager for Codex/OpenAI token storage and operations.

    Uses the generic storage and wrapper pattern for consistency.
    """

    def __init__(
        self,
        storage: GenericJsonStorage[OpenAICredentials] | None = None,
    ):
        """Initialize Codex token manager.

        Args:
            storage: Optional custom storage, defaults to standard location
        """
        if storage is None:
            storage = GenericJsonStorage(
                Path.home() / ".ccproxy" / "openai_credentials.json", OpenAICredentials
            )
        super().__init__(storage)
        self._profile_cache: OpenAIProfileInfo | None = None

    # ==================== Abstract Method Implementations ====================

    async def refresh_token(self, oauth_client: Any = None) -> OpenAICredentials | None:
        """Refresh the access token using the refresh token.

        Args:
            oauth_client: Deprecated - OAuth provider is now looked up from registry

        Returns:
            Updated credentials or None if refresh failed
        """
        # Get OAuth provider from registry
        registry = get_oauth_registry()
        oauth_provider = registry.get_provider("codex")
        if not oauth_provider:
            logger.error("codex_oauth_provider_not_found", category="auth")
            return None

        credentials = await self.load_credentials()
        if not credentials:
            logger.error("no_credentials_to_refresh", category="auth")
            return None

        if not credentials.refresh_token:
            logger.error("no_refresh_token_available", category="auth")
            return None

        try:
            # Use OAuth provider to refresh
            new_credentials: OpenAICredentials = (
                await oauth_provider.refresh_access_token(credentials.refresh_token)
            )

            # Preserve account_id if not in new credentials
            if not new_credentials.account_id and credentials.account_id:
                new_credentials.account_id = credentials.account_id

            # Save updated credentials
            if await self.save_credentials(new_credentials):
                logger.info(
                    "Token refreshed successfully",
                    account_id=new_credentials.account_id,
                    category="auth",
                )
                # Clear profile cache as token changed
                self._profile_cache = None
                return new_credentials

            logger.error("failed_to_save_refreshed_credentials", category="auth")
            return None

        except Exception as e:
            logger.error(
                "Token refresh failed",
                error=str(e),
                exc_info=False,
                category="auth",
            )
            return None

    def is_expired(self, credentials: OpenAICredentials) -> bool:
        """Check if credentials are expired using wrapper."""
        wrapper = OpenAITokenWrapper(credentials=credentials)
        return wrapper.is_expired

    def get_account_id(self, credentials: OpenAICredentials) -> str | None:
        """Get account ID from credentials."""
        return credentials.account_id

    def get_expiration_time(self, credentials: OpenAICredentials) -> datetime | None:
        """Get expiration time as datetime."""
        return credentials.expires_at

    # ==================== OpenAI-Specific Methods ====================

    async def get_profile(self) -> OpenAIProfileInfo | None:
        """Get user profile from JWT token.

        OpenAI doesn't provide a profile API, so we extract
        all information from the JWT token claims.

        Returns:
            OpenAIProfileInfo or None if not authenticated
        """
        if self._profile_cache:
            return self._profile_cache

        credentials = await self.load_credentials()
        if not credentials or self.is_expired(credentials):
            return None

        # Extract profile from JWT token claims
        self._profile_cache = OpenAIProfileInfo.from_token(credentials)
        return self._profile_cache

    async def get_access_token_with_refresh(
        self, oauth_client: Any = None
    ) -> str | None:
        """Get valid access token, automatically refreshing if expired.

        Args:
            oauth_client: Optional OAuth client for token refresh

        Returns:
            Access token if available and valid, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            logger.debug("no_credentials_found", category="auth")
            return None

        # Check if token is expired
        if self.is_expired(credentials):
            logger.info("openai_token_expired_attempting_refresh", category="auth")

            # Try to refresh if we have a refresh token
            if credentials.refresh_token:
                try:
                    refreshed = await self.refresh_token()
                    if refreshed:
                        logger.info(
                            "OpenAI token refreshed successfully", category="auth"
                        )
                        return refreshed.access_token
                    else:
                        logger.error("openai_token_refresh_failed", category="auth")
                        return None
                except Exception as e:
                    logger.error(
                        "Error refreshing OpenAI token", error=str(e), category="auth"
                    )
                    return None
            else:
                logger.warning(
                    "Cannot refresh OpenAI token - no refresh token available",
                    category="auth",
                )
                return None

        # Token is still valid
        return credentials.access_token

    async def get_access_token(self) -> str | None:
        """Override base method to return token even if expired.

        Will attempt refresh if expired but still returns the token
        even if refresh fails, letting the API handle authorization.

        Returns:
            Access token if available (expired or not), None only if no credentials
        """
        credentials = await self.load_credentials()
        if not credentials:
            logger.debug("no_credentials_found", category="auth")
            return None

        # Check if token is expired
        if self.is_expired(credentials):
            logger.warning(
                "OpenAI token is expired. Will attempt refresh but continue with expired token if needed.",
                category="auth",
            )

            # Try to refresh if we have a refresh token
            if credentials.refresh_token:
                try:
                    refreshed = await self.refresh_token()
                    if refreshed:
                        logger.info(
                            "OpenAI token refreshed successfully", category="auth"
                        )
                        return refreshed.access_token
                    else:
                        logger.warning(
                            "OpenAI token refresh failed, using expired token",
                            category="auth",
                        )
                except Exception as e:
                    logger.warning(
                        f"Error refreshing OpenAI token, using expired token: {e}",
                        category="auth",
                    )
            else:
                logger.warning(
                    "Cannot refresh expired OpenAI token (no refresh token), using expired token",
                    category="auth",
                )

        # Return the token regardless of expiration status
        return credentials.access_token

    async def get_legacy_profile(self) -> UserProfile | None:
        """Get user profile in legacy format for backward compatibility.

        This converts OpenAI credentials to a UserProfile for
        compatibility with the common auth interface.
        """
        credentials = await self.load_credentials()
        if not credentials:
            return None

        # Create minimal account info with what we have
        account = AccountInfo(
            uuid=credentials.account_id,
            email="",  # OpenAI doesn't expose email in tokens
            full_name=None,
            display_name=None,
        )

        return UserProfile(
            organization=None,  # OpenAI doesn't provide org info in tokens
            account=account,
        )
