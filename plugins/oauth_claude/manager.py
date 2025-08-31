"""Claude API token manager implementation for the Claude API plugin."""

from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import httpx

from ccproxy.auth.managers.base import BaseTokenManager
from ccproxy.auth.oauth.registry import get_oauth_registry
from ccproxy.auth.storage.base import TokenStorage
from ccproxy.core.logging import get_plugin_logger

from .models import ClaudeCredentials, ClaudeProfileInfo, ClaudeTokenWrapper
from .storage import ClaudeOAuthStorage


logger = get_plugin_logger()


class ClaudeApiTokenManager(BaseTokenManager[ClaudeCredentials]):
    """Manager for Claude API token storage and refresh operations.

    Uses the Claude-specific storage implementation.
    """

    def __init__(
        self,
        storage: TokenStorage[ClaudeCredentials] | None = None,
        http_client: "httpx.AsyncClient | None" = None,
    ):
        """Initialize Claude API token manager.

        Args:
            storage: Optional custom storage, defaults to standard location
            http_client: Optional HTTP client for API requests
        """
        if storage is None:
            storage = ClaudeOAuthStorage()
        super().__init__(storage)
        self._profile_cache: ClaudeProfileInfo | None = None

        # Create default HTTP client if not provided
        if http_client is None:
            import httpx

            http_client = httpx.AsyncClient()
        self.http_client = http_client

    @classmethod
    async def create(
        cls,
        storage: TokenStorage["ClaudeCredentials"] | None = None,
        http_client: "httpx.AsyncClient | None" = None,
    ) -> "ClaudeApiTokenManager":
        """Async factory that constructs the manager and preloads cached profile.

        This avoids creating event loops in __init__ and keeps initialization non-blocking.
        """
        manager = cls(storage=storage, http_client=http_client)
        await manager.preload_profile_cache()
        return manager

    async def preload_profile_cache(self) -> None:
        """Load profile from storage asynchronously if available."""
        try:
            from .storage import ClaudeProfileStorage

            profile_storage = ClaudeProfileStorage()

            # Only attempt to read if the file exists
            if profile_storage.file_path.exists():
                profile = await profile_storage.load_profile()
                if profile:
                    self._profile_cache = profile
                    logger.debug(
                        "claude_profile_loaded_from_cache",
                        account_id=profile.account_id,
                        email=profile.email,
                        category="auth",
                    )
        except Exception as e:
            # Don't fail if profile can't be loaded
            logger.debug(
                "claude_profile_cache_load_failed",
                error=str(e),
                category="auth",
            )

    # ==================== Abstract Method Implementations ====================

    async def refresh_token(self, oauth_client: Any = None) -> ClaudeCredentials | None:
        """Refresh the access token using the refresh token.

        Args:
            oauth_client: Deprecated - OAuth provider is now looked up from registry

        Returns:
            Updated credentials or None if refresh failed
        """
        # Get OAuth provider from registry
        registry = get_oauth_registry()
        oauth_provider = registry.get_provider("claude-api")
        if not oauth_provider:
            logger.error("claude_oauth_provider_not_found", category="auth")
            return None

        credentials = await self.load_credentials()
        if not credentials:
            logger.error("no_credentials_to_refresh", category="auth")
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        refresh_token = wrapper.refresh_token_value
        if not refresh_token:
            logger.error("no_refresh_token_available", category="auth")
            return None

        try:
            # Use OAuth provider to refresh
            new_credentials: ClaudeCredentials = (
                await oauth_provider.refresh_access_token(refresh_token)
            )

            # Save updated credentials
            if await self.save_credentials(new_credentials):
                logger.info("token_refreshed_successfully", category="auth")
                # Clear profile cache as token changed
                self._profile_cache = None

                # Fetch and update profile with new token
                wrapper = ClaudeTokenWrapper(credentials=new_credentials)
                access_token = wrapper.access_token_value
                if access_token:
                    try:
                        await self._fetch_and_save_profile(access_token)
                        logger.info(
                            "profile_updated_after_token_refresh", category="auth"
                        )
                    except Exception as e:
                        logger.warning(
                            "failed_to_update_profile_after_refresh",
                            error=str(e),
                            exc_info=e,
                            category="auth",
                        )
                        # Don't fail the refresh if profile update fails

                return new_credentials

            logger.error("failed_to_save_refreshed_credentials", category="auth")
            return None

        except Exception as e:
            logger.error(
                "Token refresh failed",
                error=str(e),
                exc_info=e,
                category="auth",
            )
            return None

    def is_expired(self, credentials: ClaudeCredentials) -> bool:
        """Check if credentials are expired using wrapper."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.is_expired

    def get_account_id(self, credentials: ClaudeCredentials) -> str | None:
        """Get account ID from credentials.

        Claude doesn't store account_id in tokens, would need
        to fetch from profile API.
        """
        if self._profile_cache:
            return self._profile_cache.account_id
        return None

    # ==================== Claude-Specific Methods ====================

    def get_expiration_time(self, credentials: ClaudeCredentials) -> datetime | None:
        """Get expiration time as datetime."""
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.expires_at_datetime

    async def get_access_token(self) -> str | None:
        """Get valid access token, automatically refreshing if expired.

        Returns:
            Access token if available and valid, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            logger.debug("no_credentials_found", category="auth")
            return None

        # Check if token is expired
        if self.is_expired(credentials):
            logger.info("claude_token_expired_attempting_refresh", category="auth")

            # Try to refresh if we have a refresh token
            wrapper = ClaudeTokenWrapper(credentials=credentials)
            refresh_token = wrapper.refresh_token_value
            if refresh_token:
                try:
                    refreshed = await self.refresh_token()
                    if refreshed:
                        logger.info(
                            "claude_token_refreshed_successfully", category="auth"
                        )
                        wrapper = ClaudeTokenWrapper(credentials=refreshed)
                        return wrapper.access_token_value
                    else:
                        logger.error("claude_token_refresh_failed", category="auth")
                        return None
                except Exception as e:
                    logger.error(
                        "Error refreshing Claude token", error=str(e), category="auth"
                    )
                    return None
            else:
                logger.warning(
                    "Cannot refresh Claude token - no refresh token available",
                    category="auth",
                )
                return None

        # Token is still valid
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.access_token_value

    async def get_access_token_value(self) -> str | None:
        """Get the actual access token value.

        Returns:
            Access token string if available, None otherwise
        """
        credentials = await self.load_credentials()
        if not credentials:
            return None

        if self.is_expired(credentials):
            return None

        wrapper = ClaudeTokenWrapper(credentials=credentials)
        return wrapper.access_token_value

    async def get_profile(self) -> ClaudeProfileInfo | None:
        """Get user profile from cache or API.

        Returns:
            ClaudeProfileInfo or None if not authenticated
        """
        if self._profile_cache:
            return self._profile_cache

        # Try to load from .account.json first
        from .storage import ClaudeProfileStorage

        profile_storage = ClaudeProfileStorage()
        profile = await profile_storage.load_profile()
        if profile:
            self._profile_cache = profile
            return profile

        # If not in storage, fetch from API
        credentials = await self.load_credentials()
        if not credentials or self.is_expired(credentials):
            return None

        # Get access token
        wrapper = ClaudeTokenWrapper(credentials=credentials)
        access_token = wrapper.access_token_value
        if not access_token:
            return None

        # Fetch profile from API and save
        try:
            from .config import ClaudeOAuthConfig

            config = ClaudeOAuthConfig()

            # Get OAuth provider from registry for detection service
            registry = get_oauth_registry()
            oauth_provider = registry.get_provider("claude-api")

            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            }

            # Add detection service headers if available
            if oauth_provider and hasattr(oauth_provider, "client"):
                custom_headers = oauth_provider.client.get_custom_headers()
                headers.update(custom_headers)

            # Use the injected HTTP client
            response = await self.http_client.get(
                config.profile_url,
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()

            profile_data = response.json()

            # Save to .account.json
            await profile_storage.save_profile(profile_data)

            # Parse and cache
            profile = ClaudeProfileInfo.from_api_response(profile_data)
            self._profile_cache = profile

            logger.info(
                "claude_profile_fetched_from_api",
                account_id=profile.account_id,
                email=profile.email,
                category="auth",
            )

            return profile

        except Exception as e:
            import httpx

            if isinstance(e, httpx.HTTPStatusError):
                logger.error(
                    "claude_profile_api_error",
                    status_code=e.response.status_code,
                    error=str(e),
                    exc_info=e,
                    category="auth",
                )
            else:
                logger.error(
                    "claude_profile_fetch_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=e,
                    category="auth",
                )
            return None

    async def close(self) -> None:
        """Close the HTTP client if it was created internally."""
        if self.http_client:
            await self.http_client.aclose()
