"""Copilot adapter implementation using delegation pattern."""

import asyncio
import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.request_context import RequestContext
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming

from .config import CopilotConfig
from .detection_service import CopilotDetectionService
from .format_adapter import CopilotFormatAdapter
from .oauth.provider import CopilotOAuthProvider
from .transformers.request import CopilotRequestTransformer
from .transformers.response import CopilotResponseTransformer


if TYPE_CHECKING:
    from ccproxy.services.interfaces import IMetricsCollector


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """GitHub Copilot adapter implementation using delegation pattern."""

    def __init__(
        self,
        config: CopilotConfig,
        oauth_provider: CopilotOAuthProvider | None = None,
        detection_service: CopilotDetectionService | None = None,
        metrics: "IMetricsCollector | None" = None,
        hook_manager: Any | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        """Initialize the Copilot adapter with explicit dependencies.

        Args:
            config: Plugin configuration
            oauth_provider: OAuth provider for authentication
            detection_service: GitHub CLI detection service
            metrics: Optional metrics collector
            hook_manager: Optional hook manager for emitting events
            http_client: Optional HTTP client
        """
        self.config = config
        self.oauth_provider = oauth_provider
        self.detection_service = detection_service
        self.metrics = metrics or self._create_null_metrics()
        self.hook_manager = hook_manager
        self.http_client = http_client
        self._owns_client = http_client is None

        # Initialize components
        self.format_adapter = CopilotFormatAdapter()
        self.request_transformer = CopilotRequestTransformer(config)
        self.response_transformer = CopilotResponseTransformer()
        self._initialized = False

    def _create_null_metrics(self) -> Any:
        """Create a null metrics collector for when none is provided."""

        class NullMetricsCollector:
            def collect_request_metric(self, **kwargs: Any) -> None:
                pass

            def collect_streaming_metric(self, **kwargs: Any) -> None:
                pass

            def collect_provider_metric(self, **kwargs: Any) -> None:
                pass

        return NullMetricsCollector()

    async def initialize(self) -> None:
        """Initialize the adapter."""
        if not self._initialized:
            # Initialize HTTP client if needed
            if self.http_client is None:
                timeout = httpx.Timeout(self.config.provider.request_timeout)
                self.http_client = httpx.AsyncClient(timeout=timeout)

            # Initialize detection service
            if self.detection_service:
                await self.detection_service.initialize_detection()

            self._initialized = True
            logger.debug("copilot_adapter_initialized")

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle a request to the Copilot API.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            method: HTTP method
            **kwargs: Additional arguments

        Returns:
            Response from Copilot API
        """
        await self.initialize()

        # Parse request body
        body = await request.body()
        if not body and method.upper() in ["POST", "PUT", "PATCH"]:
            raise HTTPException(status_code=400, detail="Request body is required")

        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON: {str(e)}"
            ) from e

        # Check if format conversion is needed
        needs_conversion = self._needs_format_conversion(endpoint, request_data)

        if needs_conversion:
            logger.debug("applying_openai_to_copilot_conversion")
            request_data = await self.format_adapter.adapt_request(
                request_data, "openai", "copilot"
            )

        # Get authentication token
        if not self.oauth_provider:
            raise HTTPException(
                status_code=503, detail="OAuth provider not initialized"
            )

        try:
            access_token = await self.oauth_provider.ensure_copilot_token()
        except ValueError as e:
            logger.error(
                "authentication_failed",
                error=str(e),
                exc_info=e,
            )
            raise HTTPException(
                status_code=401, detail=f"Authentication required: {str(e)}"
            ) from e

        # Get RequestContext
        from ccproxy.core.request_context import RequestContext

        request_context = RequestContext.get_current()
        if not request_context:
            raise HTTPException(status_code=500, detail="RequestContext not available")

        # Update context metadata
        request_context.metadata.update(
            {
                "provider": "copilot",
                "service_type": "copilot",
                "endpoint": endpoint.rstrip("/").split("/")[-1],
                "model": request_data.get("model", "gpt-4"),
                "stream": request_data.get("stream", False),
            }
        )

        # Prepare request
        target_url = self.request_transformer.get_target_url(endpoint)
        headers = self.request_transformer.transform_headers(
            dict(request.headers), access_token
        )
        transformed_body = json.dumps(request_data).encode("utf-8")

        logger.info(
            "copilot_request",
            method=method,
            endpoint=endpoint,
            target_url=target_url,
            model=request_data.get("model"),
            stream=request_data.get("stream", False),
            needs_conversion=needs_conversion,
        )

        try:
            # Make request to Copilot API
            if not self.http_client:
                raise HTTPException(status_code=503, detail="HTTP client not available")

            if request_data.get("stream", False):
                return await self.handle_streaming(
                    request,
                    endpoint,
                    method=method,
                    target_url=target_url,
                    headers=headers,
                    body=transformed_body,
                    request_data=request_data,
                    needs_conversion=needs_conversion,
                )
            else:
                return await self._handle_non_streaming_request(
                    method,
                    target_url,
                    headers,
                    transformed_body,
                    request_data,
                    needs_conversion,
                )

        except httpx.TimeoutException as e:
            logger.error("request_timeout", error=str(e), exc_info=e)
            raise HTTPException(status_code=408, detail="Request timed out") from e
        except httpx.HTTPError as e:
            logger.error(
                "http_error",
                error=str(e),
                status_code=getattr(e.response, "status_code", None)
                if hasattr(e, "response")
                else None,
                exc_info=e,
            )
            raise HTTPException(
                status_code=502, detail=f"Copilot API error: {str(e)}"
            ) from e

    async def _handle_non_streaming_request(
        self,
        method: str,
        target_url: str,
        headers: dict[str, str],
        body: bytes,
        request_data: dict[str, Any],
        needs_conversion: bool,
    ) -> Response:
        """Handle non-streaming request to Copilot API."""
        if self.http_client is None:
            raise RuntimeError("HTTP client not initialized")
        response = await self.http_client.request(
            method=method,
            url=target_url,
            headers=headers,
            content=body,
        )

        response.raise_for_status()

        # Parse response
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {"error": "Invalid JSON response from Copilot API"}

        # Convert response format if needed
        if needs_conversion:
            logger.debug("applying_copilot_to_openai_conversion")
            response_data = await self.format_adapter.adapt_response(
                response_data, "copilot", "openai"
            )

        # Transform response
        response_headers = self.response_transformer.transform_headers(
            dict(response.headers), response.status_code
        )
        response_body = json.dumps(response_data).encode("utf-8")

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle streaming request to Copilot API."""
        # Extract additional kwargs
        method = kwargs.get("method", "POST")
        target_url = kwargs.get("target_url")
        headers = kwargs.get("headers", {})
        body = kwargs.get("body", b"")
        request_data = kwargs.get("request_data", {})
        needs_conversion = kwargs.get("needs_conversion", False)

        if not target_url:
            raise ValueError("target_url is required for streaming")

        async def stream_generator() -> AsyncIterator[bytes]:
            """Generate SSE stream from Copilot API."""
            try:
                if self.http_client is None:
                    raise RuntimeError("HTTP client not initialized")
                async with self.http_client.stream(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body,
                ) as response:
                    response.raise_for_status()

                    async for chunk in response.aiter_text():
                        if chunk.strip():
                            # Parse SSE chunk
                            for line in chunk.split("\n"):
                                if line.startswith("data: "):
                                    data_content = line[6:]  # Remove 'data: '

                                    if data_content.strip() == "[DONE]":
                                        yield b"data: [DONE]\n\n"
                                        continue

                                    try:
                                        chunk_data = json.loads(data_content)

                                        # Convert format if needed
                                        if needs_conversion:
                                            chunk_data = await self.format_adapter.adapt_stream_chunk(
                                                chunk_data, "copilot", "openai"
                                            )

                                        yield f"data: {json.dumps(chunk_data)}\n\n".encode()

                                    except json.JSONDecodeError:
                                        # Pass through non-JSON chunks
                                        yield line.encode() + b"\n"

            except asyncio.CancelledError:
                logger.warning("streaming_cancelled")
                raise
            except httpx.HTTPError as e:
                logger.error("streaming_http_error", error=str(e), exc_info=e)
                error_chunk = {"error": f"HTTP error: {e}"}
                yield f"data: {json.dumps(error_chunk)}\n\n".encode()
            except Exception as e:
                logger.error("streaming_error", error=str(e), exc_info=e)
                error_chunk = {"error": str(e)}
                yield f"data: {json.dumps(error_chunk)}\n\n".encode()

        # Get streaming headers
        streaming_headers = self.response_transformer.transform_streaming_headers()

        return StreamingResponse(
            content=stream_generator(),
            media_type="text/event-stream",
            headers=streaming_headers,
        )

    def _needs_format_conversion(
        self, endpoint: str, request_data: dict[str, Any]
    ) -> bool:
        """Determine if format conversion is needed for this request.

        Args:
            endpoint: API endpoint
            request_data: Request data

        Returns:
            True if format conversion is needed
        """
        # Check if this is an OpenAI-style endpoint
        openai_endpoints = [
            "/v1/chat/completions",
            "/v1/completions",
            "/v1/embeddings",
            "/v1/models",
        ]

        for openai_endpoint in openai_endpoints:
            if endpoint.endswith(openai_endpoint):
                return True

        return False

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            if self._owns_client and self.http_client:
                await self.http_client.aclose()
                self.http_client = None

            if self.oauth_provider:
                await self.oauth_provider.cleanup()

            self._initialized = False
            logger.debug("copilot_adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "copilot_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )

    async def close(self) -> None:
        """Compatibility method - delegates to cleanup()."""
        await self.cleanup()

    # BaseHTTPAdapter abstract methods
    async def _extract_provider_auth_data(self) -> dict[str, Any]:
        """Extract provider-specific authentication data."""
        if not self.oauth_provider:
            raise ValueError("OAuth provider not available")

        access_token = await self.oauth_provider.ensure_copilot_token()
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "provider": "copilot",
        }

    async def _create_handler_config(
        self,
        needs_conversion: bool,
        request_context: Any | None = None,
    ) -> Any:
        """Create handler configuration for HTTP requests."""
        # Simple handler config for Copilot API
        return {
            "needs_conversion": needs_conversion,
            "provider": "copilot",
            "streaming_enabled": True,
        }

    async def _resolve_endpoint(self, endpoint: str) -> tuple[str, bool]:
        """Resolve target URL and determine if format conversion is needed."""
        # Use request transformer to get target URL
        target_url = self.request_transformer.get_target_url(endpoint)
        needs_conversion = self._needs_format_conversion(endpoint, {})
        return target_url, needs_conversion

    async def _update_request_context(
        self,
        request_context: RequestContext,
        endpoint: str,
        request_data: dict[str, Any],
        is_streaming: bool,
        needs_conversion: bool,
    ) -> None:
        """Update request context with Copilot-specific metadata."""
        if hasattr(request_context, "metadata"):
            request_context.metadata.update(
                {
                    "provider": "copilot",
                    "service_type": "copilot",
                    "endpoint": endpoint.rstrip("/").split("/")[-1],
                    "model": request_data.get("model", "gpt-4"),
                    "stream": request_data.get("stream", False),
                }
            )

    def _log_request(
        self,
        endpoint: str,
        request_context: RequestContext,
        is_streaming: bool,
        needs_conversion: bool,
        target_url: str,
    ) -> None:
        """Log request details."""
        logger.info(
            "copilot_request",
            endpoint=endpoint,
            target_url=target_url,
            is_streaming=is_streaming,
            needs_conversion=needs_conversion,
            has_context=request_context is not None,
        )

    async def _should_buffer_stream(
        self, request_data: dict[str, Any], is_streaming: bool
    ) -> bool:
        """Determine if stream should be buffered."""
        # Don't buffer Copilot streams by default
        return False

    async def _calculate_cost_for_usage(self, request_context: Any) -> None:
        """Calculate cost for usage (placeholder for future pricing integration)."""
        # TODO: Implement pricing calculation when pricing service is integrated
        logger.debug("copilot_cost_calculation_skipped")

    async def _wrap_streaming_response(
        self, response: StreamingResponse, request_context: Any
    ) -> StreamingResponse:
        """Wrap streaming response for metrics and logging."""
        # For now, return response as-is
        # TODO: Add metrics extraction and chunk accumulation
        return response
