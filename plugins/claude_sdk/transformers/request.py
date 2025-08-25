"""Request transformer for Claude SDK plugin.

This module handles Claude SDK-specific request transformations,
including headers and body modifications.
"""

from typing import Any

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class ClaudeSDKRequestTransformer:
    """Transform requests for Claude SDK operations.

    This transformer handles SDK-specific request modifications,
    but since the SDK handles most operations internally,
    minimal transformation is needed.
    """

    def __init__(self) -> None:
        """Initialize the request transformer."""
        self.logger = logger

    def transform_headers(
        self, headers: dict[str, str], **kwargs: Any
    ) -> dict[str, str]:
        """Transform request headers for Claude SDK.

        The SDK handles authentication internally, so we don't need
        to add API keys or other auth headers.

        Args:
            headers: Original request headers
            **kwargs: Additional context (session_id, etc.)

        Returns:
            Transformed headers
        """
        # Remove any existing auth headers since SDK handles auth
        transformed = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ["authorization", "x-api-key", "anthropic-version"]
        }

        # Add SDK-specific headers if needed
        transformed["X-Claude-SDK"] = "true"

        # Add session ID if provided
        session_id = kwargs.get("session_id")
        if session_id:
            transformed["X-Session-ID"] = session_id

        self.logger.debug(
            "claude_sdk_request_headers_transformed",
            original_count=len(headers),
            transformed_count=len(transformed),
            session_id=session_id,
            category="http",
        )

        return transformed

    def transform_body(self, body: bytes | None) -> bytes | None:
        """Transform request body for Claude SDK.

        The SDK expects specific message formats, but most transformation
        is handled by the handler and format adapters.

        Args:
            body: Original request body

        Returns:
            Transformed body (usually unchanged)
        """
        # Body transformation is handled by format adapters
        # and the handler's message conversion logic
        return body
