"""Response transformer for GitHub Copilot API."""

from typing import Any

from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class CopilotResponseTransformer:
    """Transform responses from GitHub Copilot API."""

    def __init__(self, cors_settings: dict[str, Any] | None = None):
        """Initialize response transformer.

        Args:
            cors_settings: CORS settings for response headers
        """
        self.cors_settings = cors_settings or {}

    def transform_headers(
        self, headers: dict[str, str], status_code: int = 200, **kwargs: Any
    ) -> dict[str, str]:
        """Transform response headers from Copilot API.

        Args:
            headers: Original response headers
            status_code: HTTP status code
            **kwargs: Additional context

        Returns:
            Transformed headers
        """
        # Start with original headers
        transformed = dict(headers)

        # Add CORS headers if configured
        if self.cors_settings:
            if "allow_origin" in self.cors_settings:
                transformed["access-control-allow-origin"] = self.cors_settings[
                    "allow_origin"
                ]
            if "allow_methods" in self.cors_settings:
                transformed["access-control-allow-methods"] = self.cors_settings[
                    "allow_methods"
                ]
            if "allow_headers" in self.cors_settings:
                transformed["access-control-allow-headers"] = self.cors_settings[
                    "allow_headers"
                ]

        # Ensure proper content type for JSON responses
        if "content-type" not in {k.lower() for k in transformed}:
            if status_code < 400:
                transformed["content-type"] = "application/json"
            else:
                transformed["content-type"] = "application/json"  # Errors are also JSON

        # Remove headers that might cause issues in proxying
        # Following Codex pattern - exclude headers that can cause Content-Length mismatches
        headers_to_remove = [
            "content-length",  # Will be recalculated by HTTP adapter
            "transfer-encoding",
            "content-encoding",  # May affect body length calculation
            "connection",
            "content-length",
        ]
        for header in headers_to_remove:
            for key in list(transformed.keys()):
                if key.lower() == header:
                    del transformed[key]

        logger.debug(
            "response_headers_transformed",
            original_count=len(headers),
            transformed_count=len(transformed),
            status_code=status_code,
        )

        return transformed

    def transform_body(
        self, body: bytes | str | dict[str, Any], status_code: int = 200, **kwargs: Any
    ) -> bytes:
        """Transform response body from Copilot API.

        Args:
            body: Original response body
            status_code: HTTP status code
            **kwargs: Additional context

        Returns:
            Transformed body as bytes
        """
        # For most cases, pass through the body as-is
        # Format conversion is handled by format adapters

        if isinstance(body, dict):
            import json

            transformed_body = json.dumps(body).encode("utf-8")
        elif isinstance(body, str):
            transformed_body = body.encode("utf-8")
        elif isinstance(body, bytes):
            transformed_body = body
        else:
            logger.warning(
                "unexpected_response_body_type",
                body_type=type(body).__name__,
                status_code=status_code,
            )
            transformed_body = str(body).encode("utf-8")

        logger.debug(
            "response_body_transformed",
            original_size=len(body) if isinstance(body, bytes | str) else None,
            transformed_size=len(transformed_body),
            status_code=status_code,
        )

        return transformed_body

    def transform_error_response(
        self, error: Exception, status_code: int = 500, **kwargs: Any
    ) -> tuple[dict[str, str], bytes]:
        """Transform error into proper Copilot API error response.

        Args:
            error: Exception that occurred
            status_code: HTTP status code for error
            **kwargs: Additional context

        Returns:
            Tuple of (headers, body) for error response
        """
        import json

        from ..models import CopilotError, CopilotErrorResponse

        # Create standardized error response
        if hasattr(error, "response") and hasattr(error.response, "json"):
            # If it's an HTTP error with JSON response, try to extract Copilot error
            try:
                error_data = error.response.json()
                if "error" in error_data:
                    copilot_error = CopilotError.model_validate(error_data["error"])
                else:
                    copilot_error = CopilotError(
                        message=str(error),
                        type="api_error",
                        param=None,
                        code=str(status_code),
                    )
            except Exception:
                copilot_error = CopilotError(
                    message=str(error),
                    type="api_error",
                    param=None,
                    code=str(status_code),
                )
        else:
            # Generic error
            copilot_error = CopilotError(
                message=str(error),
                type="internal_error" if status_code >= 500 else "client_error",
                param=None,
                code=str(status_code),
            )

        error_response = CopilotErrorResponse(error=copilot_error)

        # Create headers (Content-Length will be calculated by HTTP adapter)
        headers = self.transform_headers(
            {"Content-Type": "application/json"},
            status_code=status_code,
        )

        # Create body
        body = json.dumps(error_response.model_dump()).encode("utf-8")

        logger.debug(
            "error_response_transformed",
            error_type=type(error).__name__,
            status_code=status_code,
            message_length=len(copilot_error.message),
        )

        return headers, body

    def transform_streaming_headers(self, **kwargs: Any) -> dict[str, str]:
        """Get headers for streaming responses.

        Args:
            **kwargs: Additional context

        Returns:
            Headers for streaming response
        """
        headers = {
            "content-type": "text/event-stream",
            "cache-control": "no-cache",
            "connection": "keep-alive",
        }

        # Add CORS headers if configured
        if self.cors_settings:
            if "allow_origin" in self.cors_settings:
                headers["access-control-allow-origin"] = self.cors_settings[
                    "allow_origin"
                ]

        logger.debug("streaming_headers_prepared")

        return headers

    def prepare_response_context(
        self,
        headers: dict[str, str],
        body: bytes | str | dict[str, Any],
        status_code: int = 200,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare complete response context.

        Args:
            headers: Response headers
            body: Response body
            status_code: HTTP status code
            **kwargs: Additional context

        Returns:
            Response context dictionary
        """
        transformed_headers = self.transform_headers(headers, status_code, **kwargs)
        transformed_body = self.transform_body(body, status_code, **kwargs)

        context = {
            "status_code": status_code,
            "headers": transformed_headers,
            "body": transformed_body,
            "provider": "copilot",
        }

        # Add optional context
        context.update(kwargs)

        logger.debug(
            "response_context_prepared",
            status_code=status_code,
            body_size=len(transformed_body),
            header_count=len(transformed_headers),
        )

        return context
