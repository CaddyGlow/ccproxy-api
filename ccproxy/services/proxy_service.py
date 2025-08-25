"""Refactored ProxyService - orchestrates proxy requests using injected services."""

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

import httpx
import structlog
from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.config.settings import Settings
from ccproxy.core.http import BaseProxyClient
from ccproxy.hooks import HookEvent, HookManager
from ccproxy.hooks.base import HookContext
from ccproxy.observability.metrics import PrometheusMetrics
from ccproxy.services.cache import ResponseCache
from ccproxy.services.config import ProxyConfiguration
from ccproxy.services.http.connection_pool import ConnectionPoolManager
from ccproxy.services.mocking import MockResponseHandler
from ccproxy.services.streaming import StreamingHandler
from ccproxy.services.tracing import CoreRequestTracer
from ccproxy.streaming.deferred_streaming import DeferredStreaming


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext

logger = structlog.get_logger(__name__)


class ProxyService:
    """Orchestrates proxy requests using injected services."""

    def __init__(
        self,
        # Core dependencies
        proxy_client: BaseProxyClient,
        settings: Settings,
        # Injected services
        request_tracer: CoreRequestTracer,
        mock_handler: MockResponseHandler,
        streaming_handler: StreamingHandler,
        config: ProxyConfiguration,
        http_client: httpx.AsyncClient,  # Shared HTTP client for centralized management
        metrics: PrometheusMetrics | None = None,
        response_cache: ResponseCache | None = None,
        connection_pool_manager: ConnectionPoolManager | None = None,
        hook_manager: HookManager | None = None,
    ) -> None:
        """Initialize with all dependencies injected.

        - No service creation inside __init__
        - All dependencies passed from container
        - Stores references only
        """
        # Core dependencies
        self.proxy_client = proxy_client
        self.settings = settings

        # Injected services
        self.request_tracer = request_tracer
        self.mock_handler = mock_handler
        self.streaming_handler = streaming_handler
        self.config = config
        self.metrics = metrics

        # Performance optimization services
        self.response_cache = response_cache or ResponseCache()
        self.connection_pool_manager = (
            connection_pool_manager or ConnectionPoolManager()
        )

        # Shared HTTP client (injected for centralized management)
        self.http_client = http_client

        # Hook system
        self.hook_manager = hook_manager

        logger.debug(
            "ProxyService initialized with injected services and performance optimizations",
            category="lifecycle",
        )

    def set_hook_manager(self, hook_manager: HookManager) -> None:
        """Set the hook manager.

        This method allows setting the hook manager after initialization
        since the hook system is initialized after the proxy service.
        """
        self.hook_manager = hook_manager
        logger.debug("hook_manager_set_on_proxy_service", category="lifecycle")

    async def handle_request(
        self,
        request: Request,
        endpoint: str,
        method: str,
        provider: str,
        plugin_name: str,
        adapter_handler: Callable[
            ..., Awaitable[Response | StreamingResponse | DeferredStreaming]
        ],
        **kwargs: Any,
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle proxy request with hooks.

        This method provides a central point for all provider requests with hook emission.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            method: HTTP method
            provider: Provider name (e.g., 'claude_api', 'codex')
            plugin_name: Plugin name for context
            adapter_handler: The adapter's handle_request method to delegate to
            **kwargs: Additional arguments to pass to adapter

        Returns:
            Response or StreamingResponse from the adapter
        """
        start_time = time.time()

        # Get or create RequestContext
        from ccproxy.observability.context import RequestContext

        request_context = RequestContext.get_current()
        if not request_context:
            # Create a minimal context if not available
            import uuid

            request_context = RequestContext(
                request_id=str(uuid.uuid4()),
                start_time=start_time,
                logger=logger,
                metadata={
                    "endpoint": endpoint,
                    "method": method,
                    "provider": provider,
                    "plugin": plugin_name,
                },
            )

        # Emit request started hook with context
        if self.hook_manager:
            await self._emit_hook_with_context(
                HookEvent.REQUEST_STARTED,
                request_context,
                request=request,
                provider=provider,
                plugin=plugin_name,
                endpoint=endpoint,
                method=method,
            )

        try:
            # Delegate to the adapter's handle_request method
            response = await adapter_handler(request, endpoint, method, **kwargs)

            # Calculate duration and extract status
            duration = time.time() - start_time
            status = getattr(response, "status_code", 200)

            # Update request context with response info
            request_context.metadata.update(
                {
                    "status_code": status,
                    "duration_ms": duration * 1000,
                    "duration_seconds": duration,
                }
            )

            # Check if response is streaming and wrap if needed
            # DeferredStreaming already handles its own wrapping
            if isinstance(response, StreamingResponse) and not isinstance(
                response, DeferredStreaming
            ):
                response = await self._wrap_streaming_with_hooks(
                    response, request_context
                )

            # Emit request completed hook with context
            if self.hook_manager:
                await self._emit_hook_with_context(
                    HookEvent.REQUEST_COMPLETED,
                    request_context,
                    request=request,
                    response=response,
                    provider=provider,
                    plugin=plugin_name,
                    duration=duration,
                    status=status,
                )

            return response

        except Exception as e:
            # Update context with error info
            request_context.metadata.update(
                {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
            )

            # Emit request error hook with context
            if self.hook_manager:
                await self._emit_hook_with_context(
                    HookEvent.REQUEST_FAILED,
                    request_context,
                    request=request,
                    provider=provider,
                    plugin=plugin_name,
                    endpoint=endpoint,
                    method=method,
                    error=e,
                )
            raise

    async def initialize_plugins(self, scheduler: Any | None = None) -> None:
        """Initialize plugin system at startup.

        V2 plugins are initialized via FastAPI app lifecycle.
        This method is kept for backwards compatibility but does nothing.
        """
        # V2 plugins don't use this method
        pass

    async def get_pooled_client(
        self, base_url: str | None = None, streaming: bool = False
    ) -> httpx.AsyncClient:
        """Get a pooled HTTP client for the given configuration.

        Args:
            base_url: Base URL for the client
            streaming: Whether to use streaming configuration

        Returns:
            HTTPX AsyncClient from the pool
        """
        if streaming:
            return await self.connection_pool_manager.get_streaming_client(base_url)
        return await self.connection_pool_manager.get_client(base_url)

    async def _emit_hook_with_context(
        self,
        event: HookEvent,
        request_context: "RequestContext",
        **extra_data: Any,
    ) -> None:
        """Emit a hook event with RequestContext in metadata.

        Args:
            event: The hook event to emit
            request_context: The request context to include
            **extra_data: Additional data to pass to the hook
        """
        if not self.hook_manager:
            return

        from datetime import datetime

        # Create hook context with request context in metadata
        context = HookContext(
            event=event,
            timestamp=datetime.utcnow(),
            data={},  # Data will be in metadata for our use case
            metadata={
                "request_context": request_context,
                **extra_data,
            },
        )

        # Emit the hook event
        await self.hook_manager.emit_with_context(context)

    async def _wrap_streaming_with_hooks(
        self,
        response: StreamingResponse,
        request_context: "RequestContext",
    ) -> StreamingResponse:
        """Wrap streaming response to emit chunk events.

        Args:
            response: The streaming response to wrap
            request_context: The request context

        Returns:
            Wrapped streaming response that emits events
        """
        # Emit stream start event
        if self.hook_manager:
            await self._emit_hook_with_context(
                HookEvent.PROVIDER_STREAM_START,
                request_context,
                stream_metadata={
                    "start_time": time.time(),
                    "content_type": response.media_type,
                },
            )

        # Get the original iterator
        original_iterator = response.body_iterator

        # Create wrapped iterator
        async def wrapped_iterator() -> AsyncIterator[bytes]:
            """Wrap the stream iterator to emit events."""
            chunks_sent = 0
            total_bytes = 0

            try:
                async for chunk in original_iterator:
                    # Ensure chunk is bytes
                    if isinstance(chunk, str | memoryview):
                        chunk = (
                            chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                        )

                    chunks_sent += 1
                    total_bytes += len(chunk)

                    # Emit chunk event (optional, controlled by settings)
                    if self.hook_manager and self.settings.hooks.enable_chunk_events:
                        await self._emit_hook_with_context(
                            HookEvent.PROVIDER_STREAM_CHUNK,
                            request_context,
                            chunk_data=chunk,
                        )

                    yield chunk

            finally:
                # Emit stream end event
                if self.hook_manager:
                    await self._emit_hook_with_context(
                        HookEvent.PROVIDER_STREAM_END,
                        request_context,
                        stream_metrics={
                            "end_time": time.time(),
                            "chunks_sent": chunks_sent,
                            "total_bytes": total_bytes,
                        },
                    )

        # Create new streaming response with wrapped iterator
        return StreamingResponse(
            wrapped_iterator(),
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    async def close(self) -> None:
        """Clean up resources on shutdown.

        - Closes proxy client
        - Closes credentials manager
        - Closes connection pools
        - Does NOT close HTTP client (managed by ServiceContainer)
        """
        try:
            # V2 plugins are closed via FastAPI app lifecycle

            # Close proxy client
            if hasattr(self.proxy_client, "close"):
                await self.proxy_client.close()

            # Close connection pools
            if self.connection_pool_manager:
                await self.connection_pool_manager.close_all()

            # Clear response cache
            if self.response_cache:
                self.response_cache.clear()

            logger.info("proxy_service_cleanup_complete", category="lifecycle")

        except (AttributeError, TypeError) as e:
            logger.error(
                "cleanup_attribute_error",
                error=str(e),
                exc_info=e,
                category="lifecycle",
            )
        except Exception as e:
            logger.error(
                "error_during_cleanup", error=str(e), exc_info=e, category="lifecycle"
            )
