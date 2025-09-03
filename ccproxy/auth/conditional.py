"""Conditional authentication dependencies."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ccproxy.api.dependencies import SettingsDep
from ccproxy.auth.bearer import BearerTokenAuthManager
from ccproxy.auth.exceptions import AuthenticationError
from ccproxy.auth.manager import AuthManager


# FastAPI security scheme for bearer tokens
bearer_scheme = HTTPBearer(auto_error=False)


async def get_conditional_auth_manager(
    request: Request,
    settings: SettingsDep,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None, Depends(bearer_scheme)
    ] = None,
) -> AuthManager | None:
    """Get authentication manager only if auth is required.

    This dependency checks if authentication is configured and validates
    the token if required. If no auth is configured, returns None.

    Args:
        request: The FastAPI request object
        credentials: HTTP authorization credentials
        settings: Application settings

    Returns:
        AuthManager instance if authenticated, None if no auth required

    Raises:
        HTTPException: If auth is required but credentials are invalid
    """
    # Check if auth is required for this configuration
    if settings is None or not settings.security.auth_token:
        # No auth configured, return None
        return None

    # Auth is required, validate credentials
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate the token
    if credentials.credentials != settings.security.auth_token.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create and return auth manager
    try:
        bearer_auth = BearerTokenAuthManager(credentials.credentials)
        if await bearer_auth.is_authenticated():
            return bearer_auth
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except (AuthenticationError, ValueError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


# Type alias for conditional auth dependency
ConditionalAuthDep = Annotated[
    AuthManager | None, Depends(get_conditional_auth_manager)
]
