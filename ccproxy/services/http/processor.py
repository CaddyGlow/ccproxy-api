"""Request/response processing utilities for HTTP handlers."""

import json
from typing import Any

import structlog

from ccproxy.core.logging import TraceBoundLogger, get_logger
from ccproxy.services.handler_config import HandlerConfig


logger = get_logger(__name__)


class RequestProcessor:
    """Processes requests through adapters and transformers."""

    def __init__(
        self,
        logger: structlog._generic.BoundLogger
        | structlog.stdlib.BoundLogger
        | TraceBoundLogger
        | None = None,
    ) -> None:
        """Initialize the request processor.

        Args:
            logger: Optional structured logger instance
        """
        self.logger = logger or get_logger(__name__)

    async def process_request(
        self,
        body: bytes,
        headers: dict[str, str],
        handler_config: HandlerConfig,
        **transform_kwargs: Any,
    ) -> tuple[bytes, dict[str, str], bool]:
        """Process request through adapters and transformers.

        Args:
            body: Request body
            headers: Request headers
            handler_config: Handler configuration
            **transform_kwargs: Additional transformer arguments

        Returns:
            Tuple of (processed_body, processed_headers, is_streaming)
        """
        # Get logger with current request context at the start of the function
        logger = get_logger(__name__)
        self.logger = logger

        # Parse body and check streaming flag
        request_data, is_streaming = self._parse_request_body(body)

        # Apply request adapter
        processed_body = await self._apply_request_adapter(
            body, request_data, handler_config
        )

        # Filter out internal headers that shouldn't be sent upstream
        filtered_headers = self._filter_internal_headers(headers)

        # Apply request transformer
        processed_headers = self._apply_request_transformer(
            filtered_headers, handler_config, **transform_kwargs
        )

        # Transform body if transformer has that capability
        if handler_config.request_transformer and hasattr(
            handler_config.request_transformer, "transform_body"
        ):
            self.logger.debug(
                "applying_body_transformer",
                has_body=processed_body is not None,
                body_length=len(processed_body) if processed_body else 0,
                category="request",
            )
            processed_body = handler_config.request_transformer.transform_body(
                processed_body
            )
            self.logger.debug(
                "body_transformer_applied",
                body_length=len(processed_body) if processed_body else 0,
                category="request",
            )

        return processed_body, processed_headers, is_streaming

    async def process_response(
        self,
        body: bytes,
        headers: dict[str, str],
        status_code: int,
        handler_config: HandlerConfig,
        request_headers: dict[str, str] | None = None,
    ) -> tuple[bytes, dict[str, str]]:
        """Process response through adapters and transformers.

        Args:
            body: Response body
            headers: Response headers
            status_code: HTTP status code
            handler_config: Handler configuration
            request_headers: Original request headers for CORS processing

        Returns:
            Tuple of (processed_body, processed_headers)
        """
        # Apply response adapter for successful responses
        processed_body = body
        if handler_config.response_adapter and status_code < 400:
            processed_body = await self._apply_response_adapter(body, handler_config)

        # Apply response transformer with request headers for CORS
        processed_headers = self._apply_response_transformer(
            headers, handler_config, request_headers=request_headers
        )

        return processed_body, processed_headers

    def _parse_request_body(self, body: bytes) -> tuple[dict[str, Any], bool]:
        """Parse request body to extract data and streaming flag.

        Args:
            body: Request body

        Returns:
            Tuple of (parsed_data, is_streaming)
        """
        if not body:
            return {}, False

        try:
            data = json.loads(body)
            is_streaming = bool(data.get("stream", False))
            return data, is_streaming
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}, False

    async def _apply_request_adapter(
        self,
        body: bytes,
        request_data: dict[str, Any],
        handler_config: HandlerConfig,
    ) -> bytes:
        """Apply request adapter if configured.

        Args:
            body: Original request body
            request_data: Parsed request data
            handler_config: Handler configuration

        Returns:
            Adapted request body
        """
        if not handler_config.request_adapter or not body:
            return body

        try:
            adapted_data = await handler_config.request_adapter.adapt_request(
                request_data
            )
            return json.dumps(adapted_data).encode()
        except Exception as e:
            self.logger.warning(
                "request_adaptation_error",
                error=str(e),
                exc_info=e,
            )
            return body

    async def _apply_response_adapter(
        self, body: bytes, handler_config: HandlerConfig
    ) -> bytes:
        """Apply response adapter if configured.

        Args:
            body: Original response body
            handler_config: Handler configuration

        Returns:
            Adapted response body
        """
        if not handler_config.response_adapter:
            return body

        try:
            response_data = json.loads(body)
            adapted_data = await handler_config.response_adapter.adapt_response(
                response_data
            )
            return json.dumps(adapted_data).encode()
        except Exception as e:
            self.logger.warning(
                "response_adaptation_error",
                error=str(e),
                exc_info=e,
            )
            return body

    def _filter_internal_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Filter out internal headers that shouldn't be sent upstream.

        Args:
            headers: Original request headers

        Returns:
            Filtered headers dictionary
        """
        # List of headers to exclude from upstream requests
        # These are either internal tracking headers or headers that will be
        # replaced by the plugin transformers (but we remove them here as a safety net)
        internal_headers = {
            "x-request-id",
            "x-correlation-id",
            "x-ccproxy-internal",
            "user-agent",  # Will be replaced by plugin transformer, but filter here for safety
            # Add more internal headers as needed
        }

        # Filter headers (case-insensitive comparison)
        filtered = {
            key: value
            for key, value in headers.items()
            if key.lower() not in internal_headers
        }

        # Log if we filtered any headers
        removed_headers = set(headers.keys()) - set(filtered.keys())
        if removed_headers:
            self.logger.debug(
                "filtered_internal_headers",
                removed=list(removed_headers),
                count=len(removed_headers),
                category="request",
            )

        return filtered

    def _apply_request_transformer(
        self,
        headers: dict[str, str],
        handler_config: HandlerConfig,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Apply request transformer if configured.

        Args:
            headers: Request headers
            handler_config: Handler configuration
            **kwargs: Additional transformer arguments

        Returns:
            Transformed headers
        """
        self.logger.debug(
            "apply_request_transformer_start",
            has_transformer=handler_config.request_transformer is not None,
            transformer_type=type(handler_config.request_transformer).__name__
            if handler_config.request_transformer
            else None,
            kwargs_keys=list(kwargs.keys()),
            header_count=len(headers),
            category="request",
        )

        if not handler_config.request_transformer:
            self.logger.debug(
                "apply_request_transformer_no_transformer", category="request"
            )
            return headers

        if not hasattr(handler_config.request_transformer, "transform_headers"):
            self.logger.debug("apply_request_transformer_no_method", category="request")
            return headers

        try:
            # Try with kwargs first
            self.logger.debug(
                "apply_request_transformer_calling_with_kwargs",
                kwargs_keys=list(kwargs.keys()),
                category="request",
            )
            transformed = handler_config.request_transformer.transform_headers(
                headers, **kwargs
            )
            self.logger.debug(
                "apply_request_transformer_success",
                original_count=len(headers),
                transformed_count=len(transformed) if transformed else 0,
                has_auth="authorization" in (transformed or {}),
                category="request",
            )
            return transformed if transformed else headers
        except TypeError as te:
            # Fallback to no kwargs if transformer doesn't accept them
            self.logger.debug(
                "apply_request_transformer_retry_without_kwargs",
                error=str(te),
            )
            try:
                transformed = handler_config.request_transformer.transform_headers(
                    headers
                )
                self.logger.debug(
                    "apply_request_transformer_success_without_kwargs",
                    original_count=len(headers),
                    transformed_count=len(transformed) if transformed else 0,
                    has_auth="authorization" in (transformed or {}),
                )
                return transformed if transformed else headers
            except Exception as e:
                self.logger.warning(
                    "request_header_transform_error",
                    error=str(e),
                    exc_info=e,
                )
                return headers

    def _apply_response_transformer(
        self,
        headers: dict[str, str],
        handler_config: HandlerConfig,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Apply response transformer if configured.

        Args:
            headers: Response headers
            handler_config: Handler configuration
            **kwargs: Additional arguments to pass to transformer (e.g., request_headers)

        Returns:
            Transformed headers
        """
        if not handler_config.response_transformer:
            return headers

        if hasattr(handler_config.response_transformer, "transform_headers"):
            try:
                transformed = handler_config.response_transformer.transform_headers(
                    headers, **kwargs
                )
                return transformed if transformed else headers
            except Exception as e:
                self.logger.warning(
                    "response_header_transform_error",
                    error=str(e),
                    exc_info=e,
                )
        elif callable(handler_config.response_transformer):
            try:
                transformed = handler_config.response_transformer(headers)
                return transformed if transformed else headers
            except Exception as e:
                self.logger.warning(
                    "response_header_transform_callable_error",
                    error=str(e),
                    exc_info=e,
                )

        return headers
