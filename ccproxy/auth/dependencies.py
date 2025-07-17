"""FastAPI dependency injection for authentication."""

from typing import TYPE_CHECKING, Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


if TYPE_CHECKING:
    from ccproxy.config.settings import Settings

from ccproxy.auth.bearer import BearerTokenAuthManager
from ccproxy.auth.credentials_adapter import CredentialsAuthManager
from ccproxy.auth.exceptions import AuthenticationError, AuthenticationRequiredError
from ccproxy.auth.manager import AuthManager


# FastAPI security scheme for bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


async def get_credentials_auth_manager() -> AuthManager:
    """Get credentials-based authentication manager.

    Returns:
        CredentialsAuthManager instance
    """
    return CredentialsAuthManager()


async def get_bearer_auth_manager(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> AuthManager:
    """Get bearer token authentication manager.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        BearerTokenAuthManager instance

    Raises:
        HTTPException: If no valid bearer token provided
    """
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return BearerTokenAuthManager(credentials.credentials)


async def _get_auth_manager_with_settings(
    credentials: HTTPAuthorizationCredentials | None,
    settings: "Settings",
) -> AuthManager:
    """Internal function to get auth manager with specific settings.

    Args:
        credentials: HTTP authorization credentials
        settings: Application settings

    Returns:
        AuthManager instance

    Raises:
        HTTPException: If no valid authentication available
    """
    # Try bearer token first if provided
    if credentials and credentials.credentials:
        try:
            # If API has configured auth_token, validate against it
            if settings.security.auth_token:
                if credentials.credentials == settings.security.auth_token:
                    bearer_auth = BearerTokenAuthManager(credentials.credentials)
                    if await bearer_auth.is_authenticated():
                        return bearer_auth
                else:
                    # Token doesn't match configured auth_token
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid bearer token",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
            else:
                # No auth_token configured, accept any bearer token
                bearer_auth = BearerTokenAuthManager(credentials.credentials)
                if await bearer_auth.is_authenticated():
                    return bearer_auth
        except (AuthenticationError, ValueError):
            pass

    # Fall back to credentials only if no auth_token is configured
    if not settings.security.auth_token:
        try:
            credentials_auth = CredentialsAuthManager()
            if await credentials_auth.is_authenticated():
                return credentials_auth
        except AuthenticationError:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_auth_manager(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthManager:
    """Get authentication manager with fallback strategy.

    Try bearer token first, then fall back to credentials.

    Args:
        credentials: HTTP authorization credentials

    Returns:
        AuthManager instance

    Raises:
        HTTPException: If no valid authentication available
    """
    # Import here to avoid circular imports
    from ccproxy.config.settings import get_settings

    settings = get_settings()
    return await _get_auth_manager_with_settings(credentials, settings)


async def get_auth_manager_with_injected_settings(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthManager:
    """Get authentication manager with dependency-injected settings.

    This version uses FastAPI's dependency injection for settings,
    which allows test overrides to work properly.

    Args:
        credentials: HTTP authorization credentials
        settings: Application settings (injected by FastAPI)

    Returns:
        AuthManager instance

    Raises:
        HTTPException: If no valid authentication available
    """
    # Import here to avoid circular imports
    from ccproxy.config.settings import get_settings

    settings = get_settings()
    return await _get_auth_manager_with_settings(credentials, settings)


async def require_auth(
    auth_manager: Annotated[AuthManager, Depends(get_auth_manager)],
) -> AuthManager:
    """Require authentication for endpoint.

    Args:
        auth_manager: Authentication manager

    Returns:
        AuthManager instance

    Raises:
        HTTPException: If authentication fails
    """
    try:
        if not await auth_manager.is_authenticated():
            raise AuthenticationRequiredError("Authentication required")
        return auth_manager
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_access_token(
    auth_manager: Annotated[AuthManager, Depends(require_auth)],
) -> str:
    """Get access token from authenticated manager.

    Args:
        auth_manager: Authentication manager

    Returns:
        Access token string

    Raises:
        HTTPException: If token retrieval fails
    """
    try:
        return await auth_manager.get_access_token()
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def get_auth_manager_dependency(
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthManager:
    """Dependency wrapper for getting auth manager with settings injection."""
    # Import here to avoid circular imports
    from ccproxy.config.settings import get_settings

    settings = get_settings()
    return await _get_auth_manager_with_settings(credentials, settings)


# Type aliases for common dependencies
AuthManagerDep = Annotated[AuthManager, Depends(get_auth_manager)]
RequiredAuthDep = Annotated[AuthManager, Depends(require_auth)]
AccessTokenDep = Annotated[str, Depends(get_access_token)]
