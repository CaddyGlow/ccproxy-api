from abc import abstractmethod
from typing import Any, cast

import httpx
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.models.provider import ProviderConfig
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import extract_request_headers, filter_response_headers


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
            body = await self._execute_format_chain(
                body, ctx.format_chain, ctx, mode="request"
            )
        # Step 3: Provider-specific preparation
        prepared_body, prepared_headers = await self.prepare_provider_request(
            body, headers, endpoint
        )

        # Step 4: Execute HTTP request
        target_url = await self.get_target_url(endpoint)
        provider_response = await self._execute_http_request(
            method,
            target_url,
            prepared_headers,
            prepared_body,
        )

        # Step 5: Provider-specific response processing
        response = await self.process_provider_response(provider_response, endpoint)

        # filter out hop-by-hop headers
        headers = filter_response_headers(dict(provider_response.headers))

        # Step 6: Format the response
        if isinstance(response, StreamingResponse):
            return await self._convert_streaming_response(
                response, ctx.format_chain, ctx
            )
        elif isinstance(response, Response):
            if ctx.format_chain and len(ctx.format_chain) > 1:
                if provider_response.status_code >= 400:
                    # Error response; use error format chain if specified
                    body_response = await self._execute_format_chain(
                        cast(bytes, response.body),
                        ctx.format_chain,
                        ctx,
                        mode="error",
                    )
                    if "content-length" in response.headers:
                        response.headers["content-length"] = str(len(body_response))
                    return Response(
                        content=body_response,
                        status_code=provider_response.status_code,
                        headers=headers,
                        media_type=provider_response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
                else:
                    body_response = await self._execute_format_chain(
                        cast(bytes, response.body),
                        ctx.format_chain,
                        ctx,
                        mode="response",
                    )
                    if "content-length" in response.headers:
                        response.headers["content-length"] = str(len(body_response))
                    return Response(
                        content=body_response,
                        status_code=provider_response.status_code,
                        headers=headers,
                        media_type=provider_response.headers.get(
                            "content-type", "application/json"
                        ),
                    )
            else:
                logger.debug("format_chain_skipped", reason="no forward chain")
                return response
        else:
            logger.warning(
                "unexpected_provider_response_type", type=type(response).__name__
            )
        return Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=headers,
            media_type=headers.get("content-type", "application/json"),
        )
        # raise ValueError(
        #     "process_provider_response must return httpx.Response for non-streaming",
        # )

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
        self, body: bytes, format_chain: list[str], ctx: Any, mode: str = "response"
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
                adapter = self.format_registry.get(from_format, to_format)
                if not adapter:
                    raise ValueError(
                        f"No adapter found for {from_format} -> {to_format}"
                    )
                if mode == "request":
                    current_data = await adapter.adapt_request(current_data)
                elif mode == "response":
                    current_data = await adapter.adapt_response(current_data)
                elif mode == "error":
                    current_data = await adapter.adapt_error(current_data)
                else:
                    raise ValueError(f"Invalid mode for format chain: {mode}")

                logger.debug(
                    "format_chain_step_completed",
                    from_format=from_format,
                    to_format=to_format,
                    mode=mode,
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

    # async def _execute_reverse_format_chain(
    #     self, response: Response | StreamingResponse, format_chain: list[str], ctx: Any
    # ) -> Response | StreamingResponse:
    #     """Execute reverse format conversion chain on responses."""
    #
    #     if not self.format_registry:
    #         logger.debug("reverse_format_chain_skipped", reason="no_registry")
    #         return response
    #
    #     # Handle streaming vs non-streaming responses
    #     if isinstance(response, StreamingResponse):
    #         return await self._convert_streaming_response(response, format_chain, ctx)
    #     else:
    #         return await self._convert_regular_response(response, format_chain, ctx)

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
