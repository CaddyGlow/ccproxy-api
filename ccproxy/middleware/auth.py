"""Authentication middleware for Claude Proxy API."""

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ccproxy.config import get_settings
from ccproxy.exceptions import AuthenticationError
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)

# HTTP Bearer scheme for extracting tokens
bearer_scheme = HTTPBearer(auto_error=False)


def extract_token_from_headers(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    request: Request,
) -> str | None:
    """
    Extract authentication token from various header formats.

    Supports:
    - Anthropic format: x-api-key: <token>
    - OpenAI/Classic format: Authorization: Bearer <token>

    Args:
        credentials: Bearer token credentials from Authorization header
        request: FastAPI request object

    Returns:
        The extracted token or None if not found
    """
    # Check x-api-key header first (Anthropic style)
    x_api_key = request.headers.get("x-api-key")
    if x_api_key:
        logger.debug("Found token in x-api-key header")
        return x_api_key

    # Check Authorization Bearer header (OpenAI/Classic style)
    if credentials and credentials.credentials:
        logger.debug("Found token in Authorization Bearer header")
        return credentials.credentials

    return None


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
    request: Request,
) -> None:
    """
    Verify authentication token from multiple header formats.

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

    # Extract token from headers
    token = extract_token_from_headers(credentials, request)

    # Check if any token was provided
    if not token:
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
    if token != settings.auth_token:
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
