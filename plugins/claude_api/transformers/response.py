"""Claude API response transformer."""

from typing import Any

from ccproxy.config.cors import CORSSettings
from ccproxy.core.logging import get_plugin_logger
from ccproxy.utils.cors import get_cors_headers, get_request_origin


logger = get_plugin_logger()


class ClaudeAPIResponseTransformer:
    """Transform responses from Claude API.

    Handles:
    - Header passthrough and filtering
    - Error response preservation
    - Server header forwarding
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

        Preserves important headers from upstream including server headers.

        Args:
            headers: Original response headers from upstream
            **kwargs: Additional arguments including request_headers for CORS

        Returns:
            Transformed headers preserving server identification and secure CORS
        """
        # Get logger with request context at the start of the function
        logger = get_plugin_logger()

        transformed = {}

        # Headers to exclude from passthrough
        excluded = {
            "content-length",  # Will be recalculated
            "transfer-encoding",  # Handled by framework
            "content-encoding",  # Decompression handled by httpx
            "connection",  # Hop-by-hop header
            "date",
        }

        # Pass through most headers, including server headers
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
        """Transform response body.

        For Claude API, we pass through the body as-is to preserve
        the exact response format, especially for error responses.

        Args:
            body: Original response body

        Returns:
            Response body (unchanged for Claude API)
        """
        # Pass through body unchanged - no transformation needed
        # This ensures error responses are preserved exactly as returned by Claude API
        return body

    async def transform(
        self, status_code: int, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict[str, str], bytes | None]:
        """Transform response components.

        Args:
            status_code: HTTP status code
            headers: Response headers
            body: Response body

        Returns:
            Tuple of (status_code, transformed_headers, transformed_body)
        """
        transformed_headers = self.transform_headers(headers)
        transformed_body = self.transform_body(body)

        # Preserve the original status code (including error codes)
        return status_code, transformed_headers, transformed_body

    async def adapt_response(self, response_json: dict[str, Any]) -> dict[str, Any]:
        """Adapt response for compatibility with ProxyService.

        This method provides the interface expected by ProxyService.
        For Claude API, we pass through responses unchanged.

        Args:
            response_json: Response body as JSON dict

        Returns:
            Response body as JSON dict (unchanged)
        """
        # Pass through response unchanged
        return response_json
