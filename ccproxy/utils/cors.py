"""CORS utilities for plugins and transformers."""

from typing import TYPE_CHECKING

import structlog


if TYPE_CHECKING:
    from ccproxy.config.core import CORSSettings

logger = structlog.get_logger(__name__)


def get_cors_headers(
    cors_settings: "CORSSettings",
    request_origin: str | None = None,
    request_headers: dict[str, str] | None = None,
) -> dict[str, str]:
    """Get CORS headers based on configuration and request.

    Args:
        cors_settings: CORS configuration settings
        request_origin: Origin from the request Origin header
        request_headers: Request headers dict for method/header validation

    Returns:
        dict: CORS headers to add to response
    """
    headers = {}

    # Handle Access-Control-Allow-Origin
    allowed_origin = cors_settings.get_allowed_origin(request_origin)
    if allowed_origin:
        headers["Access-Control-Allow-Origin"] = allowed_origin

    # Handle credentials
    if cors_settings.credentials and allowed_origin != "*":
        headers["Access-Control-Allow-Credentials"] = "true"

    # Handle methods
    if cors_settings.methods:
        # Convert list to comma-separated string
        if "*" in cors_settings.methods:
            headers["Access-Control-Allow-Methods"] = "*"
        else:
            headers["Access-Control-Allow-Methods"] = ", ".join(cors_settings.methods)

    # Handle headers
    if cors_settings.headers:
        # Convert list to comma-separated string
        if "*" in cors_settings.headers:
            headers["Access-Control-Allow-Headers"] = "*"
        else:
            headers["Access-Control-Allow-Headers"] = ", ".join(cors_settings.headers)

    # Handle exposed headers
    if cors_settings.expose_headers:
        headers["Access-Control-Expose-Headers"] = ", ".join(
            cors_settings.expose_headers
        )

    # Handle max age for preflight requests
    if cors_settings.max_age > 0:
        headers["Access-Control-Max-Age"] = str(cors_settings.max_age)

    logger.debug(
        "cors_headers_generated",
        request_origin=request_origin,
        allowed_origin=allowed_origin,
        headers_count=len(headers),
    )

    return headers


def should_handle_cors(request_headers: dict[str, str] | None) -> bool:
    """Check if request requires CORS handling.

    Args:
        request_headers: Request headers

    Returns:
        bool: True if CORS handling is needed
    """
    if not request_headers:
        return False

    # CORS is needed if Origin header is present
    return any(key.lower() == "origin" for key in request_headers)


def get_request_origin(request_headers: dict[str, str] | None) -> str | None:
    """Extract origin from request headers.

    Args:
        request_headers: Request headers

    Returns:
        str | None: Origin value or None if not present
    """
    if not request_headers:
        return None

    # Find origin header (case-insensitive)
    for key, value in request_headers.items():
        if key.lower() == "origin":
            return value

    return None
