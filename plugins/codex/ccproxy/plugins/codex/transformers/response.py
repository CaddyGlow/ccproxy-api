"""Codex response transformer - passthrough pattern."""

from typing import Any

from ccproxy.config.cors import CORSSettings
from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.cors import get_cors_headers, get_request_origin


logger = get_plugin_logger()


class CodexResponseTransformer:
    """Transform responses from Codex API.

    Handles:
    - Header filtering and CORS addition
    - Body passthrough (no transformation)
    """

    def __init__(self, cors_settings: CORSSettings | None = None) -> None:
        """Initialize the response transformer.

        Args:
            cors_settings: CORS configuration settings
        """
        self.cors_settings = cors_settings

    def transform_headers(
        self, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, str]:
        """Transform response headers.

        Args:
            headers: Original response headers
            **kwargs: Additional arguments including request_headers for CORS

        Returns:
            Filtered headers with secure CORS
        """
        transformed = {}

        # Headers to exclude
        excluded = {
            "content-length",
            "transfer-encoding",
            "content-encoding",
            "connection",
        }

        # Pass through non-excluded headers
        for key, value in headers.items():
            if key.lower() not in excluded:
                transformed[key] = value

        # Add secure CORS headers if settings are available
        if self.cors_settings:
            request_headers = kwargs.get("request_headers", {})
            request_origin = get_request_origin(request_headers)
            cors_headers = get_cors_headers(
                self.cors_settings, request_origin, request_headers
            )
            transformed.update(cors_headers)
        else:
            # Fallback to secure defaults if no CORS settings available
            logger.warning(
                "cors_settings_not_available_using_fallback", category="transform"
            )
            # Only add CORS headers if Origin header is present in request
            request_headers = kwargs.get("request_headers", {})
            request_origin = get_request_origin(request_headers)
            # Use a secure default - localhost origins only
            if request_origin and any(
                origin in request_origin for origin in ["localhost", "127.0.0.1"]
            ):
                transformed["Access-Control-Allow-Origin"] = request_origin
                transformed["Access-Control-Allow-Headers"] = (
                    "Content-Type, Authorization, Accept, Origin, X-Requested-With"
                )
                transformed["Access-Control-Allow-Methods"] = (
                    "GET, POST, PUT, DELETE, OPTIONS"
                )

        return transformed

    def transform_body(self, body: bytes | None) -> bytes | None:
        """Transform response body - passthrough.

        Args:
            body: Original response body

        Returns:
            Response body unchanged
        """
        return body
