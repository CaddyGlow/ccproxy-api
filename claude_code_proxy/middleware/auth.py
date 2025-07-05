"""Authentication middleware for Claude Proxy API."""

import logging
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from claude_code_proxy.config import get_settings
from claude_code_proxy.exceptions import AuthenticationError


logger = logging.getLogger(__name__)

# HTTP Bearer scheme for extracting tokens
bearer_scheme = HTTPBearer(auto_error=False)


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    request: Request,
) -> None:
    """
    Verify bearer token authentication.

    Args:
        credentials: Bearer token credentials from request header
        request: FastAPI request object

    Raises:
        HTTPException: If authentication fails
    """
    settings = get_settings()

    # Skip authentication if no token is configured
    if not settings.auth_token:
        logger.debug("No auth token configured, skipping authentication")
        return

    # Check if credentials are provided
    if not credentials:
        logger.warning(f"Missing authentication token for {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "authentication_error",
                    "message": "Missing authentication token",
                }
            },
        )

    # Verify token
    if credentials.credentials != settings.auth_token:
        logger.warning(f"Invalid authentication token for {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "type": "authentication_error",
                    "message": "Invalid authentication token",
                }
            },
        )

    logger.debug(f"Authentication successful for {request.url.path}")


def get_auth_dependency() -> Callable[
    [Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)], Request],
    None,
]:
    """
    Get authentication dependency function.

    Returns:
        Authentication dependency function
    """
    return verify_token
