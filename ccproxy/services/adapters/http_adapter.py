from abc import abstractmethod
from typing import Any

import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import extract_request_headers, to_canonical_headers


logger = get_plugin_logger()


class BaseHTTPAdapter(BaseAdapter):
    """Simplified HTTP adapter with format chain support."""

    def __init__(
        self, auth_manager: Any, http_pool_manager: Any, **kwargs: Any
    ) -> None:
        self.auth_manager = auth_manager
        self.http_pool_manager = http_pool_manager
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

        # Step 2: Execute format chain if specified
        if ctx.format_chain and len(ctx.format_chain) > 1:
            body = await self._execute_format_chain(body, ctx.format_chain, ctx)

        # Step 3: Provider-specific preparation
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, ctx.metadata.get("endpoint", "/")
        )

        # Step 4: Execute HTTP request
        target_url = await self.get_target_url(ctx.metadata.get("endpoint", "/"))
        response = await self._execute_http_request(
            ctx.metadata.get("method", request.method),
            target_url,
            prepared_headers,
            prepared_body,
        )

        # Step 5: Provider-specific response processing
        return await self.process_provider_response(
            response, ctx.metadata.get("endpoint", "/")
        )

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
        current_body = body
        format_registry = self.context.get("format_registry") if self.context else None

        if not format_registry:
            ctx.log_event("format_chain_skipped", reason="no_registry")
            return body

        for i in range(len(format_chain) - 1):
            from_format = format_chain[i]
            to_format = format_chain[i + 1]

            try:
                adapter = format_registry.get_adapter(from_format, to_format)
                current_body = await adapter.convert(current_body)

                ctx.log_event(
                    "format_chain_step_completed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                )

            except Exception as e:
                ctx.log_event(
                    "format_chain_step_failed",
                    from_format=from_format,
                    to_format=to_format,
                    step=i + 1,
                    error=str(e),
                )
                raise

        return current_body

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
        canonical_headers = to_canonical_headers(headers)

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
