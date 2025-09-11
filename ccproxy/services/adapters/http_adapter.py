from abc import abstractmethod
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import extract_request_headers


logger = get_plugin_logger()


class BaseHTTPAdapter(BaseAdapter):
    """Simplified HTTP adapter with format chain support."""

    def __init__(
        self,
        config: ProviderConfig,
        auth_manager: Any,
        http_pool_manager: Any,
        **kwargs: Any,
    ) -> None:
        # Call parent constructor to properly initialize config
        super().__init__(config=config, **kwargs)
        self.auth_manager = auth_manager
        self.http_pool_manager = http_pool_manager
        self.format_registry = kwargs.get("format_registry")
        self.context = kwargs.get("context")

    async def handle_request(
        self, request: Request
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle request with simplified single parameter signature."""

        # Get context from middleware (already initialized)
        ctx = request.state.context

        # Step 1: Extract request data
        body = await request.body()
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")

        # Step 2: Execute format chain if specified
        if ctx.format_chain and len(ctx.format_chain) > 1:
            body = await self._execute_format_chain(body, ctx.format_chain, ctx)

        # Step 3: Provider-specific preparation
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )

        # Step 4: Execute HTTP request
        target_url = await self.get_target_url(endpoint)
        response = await self._execute_http_request(
            method,
            target_url,
            prepared_headers,
            prepared_body,
        )

        # Step 5: Provider-specific response processing
        provider_response = await self.process_provider_response(response, endpoint)

        # Step 6: Execute reverse format chain if specified
        if ctx.format_chain and len(ctx.format_chain) > 1:
            provider_response = await self._execute_reverse_format_chain(
                provider_response, ctx.format_chain, ctx
            )

        return provider_response

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request (BaseAdapter interface).

        For HTTP adapters, streaming is handled within the main handle_request flow.
        This delegates to the main handler and ensures streaming response.
        """
        # Set endpoint in context for compatibility with BaseAdapter interface
        if hasattr(request.state, "context"):
            ctx = request.state.context
            ctx.metadata["endpoint"] = endpoint

        response = await self.handle_request(request)

        # Ensure we return a streaming response
        if isinstance(response, StreamingResponse | DeferredStreaming):
            return response
        else:
            # Convert regular response to streaming if needed
            # This shouldn't normally happen for streaming requests
            logger.warning(
                "non_streaming_response_for_streaming_request",
                endpoint=endpoint,
                response_type=type(response).__name__,
            )
            return response  # type: ignore[return-value]

    async def _execute_format_chain(
        self, body: bytes, format_chain: list[str], ctx: Any
    ) -> bytes:
        """Execute format conversion chain."""
        import json

        if not self.format_registry:
            logger.debug("format_chain_skipped", reason="no_registry")
            return body

        # Parse body as JSON for format conversion
        try:
            current_data = json.loads(body.decode()) if body else {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("format_chain_parse_failed", error=str(e))
            return body

        for i in range(len(format_chain) - 1):
            from_format = format_chain[i]
            to_format = format_chain[i + 1]

            try:
                adapter = self.format_registry.get_adapter(from_format, to_format)
                current_data = await adapter.adapt_request(current_data)

                logger.debug(
                    "format_chain_step_completed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                )

            except Exception as e:
                logger.debug(
                    "format_chain_step_failed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                    error=str(e),
                )
                raise

        # Convert back to bytes
        return json.dumps(current_data).encode()

    async def _execute_reverse_format_chain(
        self, response: Response | StreamingResponse, format_chain: list[str], ctx: Any
    ) -> Response | StreamingResponse:
        """Execute reverse format conversion chain on responses."""

        if not self.format_registry:
            logger.debug("reverse_format_chain_skipped", reason="no_registry")
            return response

        # Handle streaming vs non-streaming responses
        if isinstance(response, StreamingResponse):
            return await self._convert_streaming_response(response, format_chain, ctx)
        else:
            return await self._convert_regular_response(response, format_chain, ctx)

    async def _convert_regular_response(
        self, response: Response, format_chain: list[str], ctx: Any
    ) -> Response:
        """Convert non-streaming response through reverse format chain."""
        import json

        try:
            # Parse response body as JSON
            if response.body:
                body_str = (
                    response.body.decode()
                    if isinstance(response.body, bytes)
                    else str(response.body)
                )
                response_data = json.loads(body_str)
            else:
                response_data = {}
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.debug("reverse_format_chain_parse_failed", error=str(e))
            return response

        # Check if this is an error response based on status code
        is_error_response = response.status_code >= 400

        # Execute reverse format chain (last to first)
        current_data = response_data
        for i in range(len(format_chain) - 1, 0, -1):
            from_format = format_chain[i]
            to_format = format_chain[i - 1]

            try:
                if not self.format_registry:
                    logger.debug("reverse_format_chain_no_registry")
                    return response
                adapter = self.format_registry.get_adapter(from_format, to_format)

                # Use adapt_error for error responses, adapt_response for success
                if is_error_response and hasattr(adapter, "adapt_error"):
                    # For error responses, create a standard error structure if needed
                    if "error" not in current_data:
                        # Convert non-standard error response to standard error format
                        current_data = {
                            "error": {
                                "type": "internal_server_error"
                                if response.status_code >= 500
                                else "invalid_request_error",
                                "message": f"Request failed with status {response.status_code}",
                                "response_data": current_data,  # Preserve original response
                            }
                        }
                    logger.debug(
                        "reverse_format_chain_using_adapt_error",
                        from_format=from_format,
                        to_format=to_format,
                        status_code=response.status_code,
                    )
                    current_data = adapter.adapt_error(current_data)
                else:
                    logger.debug(
                        "reverse_format_chain_using_adapt_response",
                        from_format=from_format,
                        to_format=to_format,
                        current_data=current_data,
                    )
                    current_data = await adapter.adapt_response(current_data)
                    logger.debug(
                        "reverse_format_chain_using_adapt_response",
                        from_format=from_format,
                        to_format=to_format,
                        current_data=current_data,
                    )

                logger.debug(
                    "reverse_format_chain_step_completed",
                    from_format=from_format,
                    to_format=to_format,
                    step=len(format_chain) - i,
                )

            except Exception as e:
                logger.debug(
                    "reverse_format_chain_step_failed",
                    from_format=from_format,
                    to_format=to_format,
                    step=len(format_chain) - i,
                    error=str(e),
                )
                # Log error but continue with original response to avoid breaking
                logger.debug("reverse_format_chain_fallback_to_original")
                return response

        # Create new response with converted data
        converted_body = json.dumps(current_data).encode()

        # Ensure proper headers for JSON response
        headers = dict(response.headers) if response.headers else {}
        headers["content-type"] = "application/json"
        headers["content-length"] = str(len(converted_body))

        return Response(
            content=converted_body,
            status_code=response.status_code,
            headers=headers,
            media_type="application/json",
        )

    async def _convert_streaming_response(
        self, response: StreamingResponse, format_chain: list[str], ctx: Any
    ) -> StreamingResponse:
        """Convert streaming response through reverse format chain."""
        # For now, disable reverse format chain for streaming responses
        # This complex conversion should be handled by the existing format adapter system
        # TODO: Implement proper streaming format conversion
        logger.debug(
            "reverse_streaming_format_chain_disabled",
            reason="complex_sse_parsing_disabled",
            format_chain=format_chain,
        )
        return response

    @abstractmethod
    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        """Provider prepares request. Headers have lowercase keys."""
        pass

    @abstractmethod
    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse:
        """Provider processes response."""
        pass

    @abstractmethod
    async def get_target_url(self, endpoint: str) -> str:
        """Get target URL for this provider."""
        pass

    async def _execute_http_request(
        self, method: str, url: str, headers: dict[str, str], body: bytes
    ) -> httpx.Response:
        """Execute HTTP request."""
        # Convert to canonical headers for HTTP
        canonical_headers = headers

        # Get HTTP client
        client = await self.http_pool_manager.get_client()

        # Execute
        response: httpx.Response = await client.request(
            method=method,
            url=url,
            headers=canonical_headers,
            content=body,
            timeout=120.0,
        )
        return response

    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.debug("adapter_cleanup_completed")
