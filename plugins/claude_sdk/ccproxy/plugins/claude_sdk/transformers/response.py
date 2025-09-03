"""Response transformer for Claude SDK plugin.

This module handles Claude SDK-specific response transformations,
including headers and body modifications.
"""

from typing import TYPE_CHECKING, Any

from ccproxy.core.logging import get_plugin_logger


if TYPE_CHECKING:
    from ccproxy.config.cors import CORSSettings


logger = get_plugin_logger()


class ClaudeSDKResponseTransformer:
    """Transform responses from Claude SDK operations.

    This transformer handles SDK-specific response modifications,
    including CORS headers and SDK indicators.
    """

    def __init__(self, cors_settings: "CORSSettings | None" = None) -> None:
        """Initialize the response transformer.

        Args:
            cors_settings: CORS configuration settings
        """
        self.logger = logger
        self.cors_settings = cors_settings

    def transform_headers(
        self, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, str]:
        """Transform response headers from Claude SDK.

        Add SDK-specific headers and secure CORS headers.

        Args:
            headers: Original response headers
            **kwargs: Additional arguments including request_headers for CORS

        Returns:
            Transformed headers
        """
        transformed = headers.copy()

        # Add SDK indicator headers
        transformed["X-Claude-SDK-Response"] = "true"

        # Ensure proper content type for streaming
        if "text/event-stream" in transformed.get("content-type", ""):
            # Already set correctly for SSE
            pass
        elif "application/json" not in transformed.get("content-type", ""):
            # Default to JSON if not set
            transformed["content-type"] = "application/json"

        # Add secure CORS headers if settings are available
        if self.cors_settings:
            from ccproxy.utils.cors import get_cors_headers, get_request_origin

            request_headers = kwargs.get("request_headers", {})
            request_origin = get_request_origin(request_headers)
            cors_headers = get_cors_headers(
                self.cors_settings, request_origin, request_headers
            )
            transformed.update(cors_headers)
        else:
            # Fallback to secure defaults if no CORS settings available
            self.logger.warning(
                "cors_settings_not_available_using_fallback", category="transform"
            )
            # Only add CORS headers if Origin header is present in request
            request_headers = kwargs.get("request_headers", {})
            from ccproxy.utils.cors import get_request_origin

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

        self.logger.debug(
            "claude_sdk_response_headers_transformed",
            original_count=len(headers),
            transformed_count=len(transformed),
            category="http",
        )

        return transformed

    def transform_body(self, body: bytes | None) -> bytes | None:
        """Transform response body from Claude SDK.

        Body transformation is handled by format adapters and the
        streaming processor, so this is usually a passthrough.

        Args:
            body: Original response body

        Returns:
            Transformed body (usually unchanged)
        """
        # Body transformation is handled by format adapters
        # and the handler's response conversion logic
        return body
