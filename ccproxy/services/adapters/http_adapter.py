"""Base HTTP adapter for HTTP-based providers (claude_api and codex)."""

import json
from abc import abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.hooks import HookEvent, HookManager
from ccproxy.hooks.base import HookContext
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.services.http.plugin_handler import PluginHTTPHandler
from ccproxy.services.interfaces import (
    IMetricsCollector,
    IRequestTracer,
    IStreamingHandler,
    NullMetricsCollector,
    NullRequestTracer,
    NullStreamingHandler,
)
from ccproxy.services.streaming.buffer_service import StreamingBufferService
from ccproxy.streaming.deferred_streaming import DeferredStreaming


if TYPE_CHECKING:
    import httpx

    from ccproxy.auth.manager import AuthManager
    from ccproxy.core.plugins.declaration import PluginContext
    from ccproxy.core.request_context import RequestContext
    from ccproxy.services.handler_config import PluginTransformerProtocol
    from ccproxy.services.http_pool import HTTPPoolManager


logger = get_plugin_logger()


class BaseHTTPAdapter(BaseAdapter):
    """Base adapter class for HTTP-based providers.

    This class extracts common HTTP orchestration logic from ClaudeAPIAdapter
    and CodexAdapter to reduce duplication. It handles:
    - Request/response transformation with pluggable transformers
    - Streaming vs regular responses
    - Integration with PluginHTTPHandler for HTTP execution
    - Provider-specific customization hooks
    """

    def __init__(
        self,
        # Required dependencies
        auth_manager: "AuthManager",
        detection_service: Any,
        http_pool_manager: "HTTPPoolManager",
        # Optional dependencies with defaults
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "IStreamingHandler | None" = None,
        request_transformer: "PluginTransformerProtocol | None" = None,
        response_transformer: "PluginTransformerProtocol | None" = None,
        hook_manager: "HookManager | None" = None,
        buffer_service: "StreamingBufferService | None" = None,
        # Context for plugin-specific services
        context: "PluginContext | None" = None,
    ) -> None:
        """Initialize the base HTTP adapter with explicit dependencies.

        Args:
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for CLI detection
            http_pool_manager: HTTP pool manager for getting clients on demand
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            request_transformer: Optional request transformer
            response_transformer: Optional response transformer
            hook_manager: Optional hook manager for event emission
            buffer_service: Optional streaming buffer service for stream-to-buffer conversion
            context: Optional plugin context containing plugin_registry and other services
        """
        # Store required dependencies
        self._auth_manager = auth_manager
        self._detection_service = detection_service
        self._hook_manager = hook_manager
        self._http_pool_manager = http_pool_manager

        # Use null object pattern for optional dependencies
        self.request_tracer = request_tracer or NullRequestTracer()
        self.metrics = metrics or NullMetricsCollector()
        self.streaming_handler = streaming_handler or NullStreamingHandler()

        # Initialize buffer service (will be lazily created if None)
        self._buffer_service = buffer_service

        # Store context for plugin-specific needs
        self.context = context or {}

        # Initialize transformers (can be overridden by subclasses)
        self._request_transformer = request_transformer
        self._response_transformer = response_transformer

        # Initialize HTTP handler with explicit dependencies
        self._http_handler: PluginHTTPHandler = PluginHTTPHandler(
            request_tracer=self.request_tracer,
            http_pool_manager=http_pool_manager,
        )

    def _get_hook_manager(self) -> "HookManager | None":
        """Get the injected hook manager."""
        return self._hook_manager

    async def _get_http_client(self) -> "httpx.AsyncClient":
        """Get HTTP client from pool manager.

        Returns:
            HTTP client instance

        Raises:
            RuntimeError: If no HTTP pool manager is available
        """
        if self._http_pool_manager is not None:
            return await self._http_pool_manager.get_client()

        raise RuntimeError("HTTP pool manager is required but not available")

    def _get_pricing_service(self) -> Any:
        """Get pricing service from plugin registry if available.

        Avoid importing plugin symbols in core; fetch by name only.
        """
        try:
            if not self.context or "plugin_registry" not in self.context:
                return None

            plugin_registry = self.context["plugin_registry"]

            # Do not import plugin types in core. Retrieve by name only.
            return plugin_registry.get_service("pricing")

        except Exception as e:
            logger.debug("failed_to_get_pricing_service", error=str(e))
            return None

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Handle a request using the common HTTP flow.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            method: HTTP method
            **kwargs: Additional arguments

        Returns:
            Response from the provider API
        """
        # Validate prerequisites
        self._validate_prerequisites()

        # Get RequestContext - it must exist during the app request lifecycle
        from ccproxy.core.request_context import RequestContext

        request_context: RequestContext | None = RequestContext.get_current()
        if not request_context:
            raise HTTPException(
                status_code=500,
                detail=(
                    "RequestContext not available - plugin must be invoked within the "
                    "application request lifecycle"
                ),
            )

        # Get request body and auth
        body = await request.body()

        # Get access token directly from auth manager
        access_token = await self._auth_manager.get_access_token()

        # Build auth headers with Bearer token only if available
        auth_headers: dict[str, str] = {}
        if access_token:
            auth_headers["Authorization"] = f"Bearer {access_token}"

        # Determine endpoint handling (provider-specific)
        target_url, needs_conversion = await self._resolve_endpoint(endpoint)

        # Create handler configuration (provider-specific)
        handler_config = await self._create_handler_config(
            needs_conversion, request_context
        )

        # Prepare and execute request
        return await self._execute_request(
            method=method,
            target_url=target_url,
            body=body,
            auth_headers=auth_headers,
            access_token=access_token,
            request_headers=dict(request.headers),
            handler_config=handler_config,
            endpoint=endpoint,
            needs_conversion=needs_conversion,
            request_context=request_context,
        )

    def _validate_prerequisites(self) -> None:
        """Validate that required components are available."""
        if not self._auth_manager:
            raise HTTPException(
                status_code=503, detail="Authentication manager not available"
            )
        if not self._http_handler:
            raise HTTPException(status_code=503, detail="HTTP handler not initialized")

    @abstractmethod
    async def _resolve_endpoint(self, endpoint: str) -> tuple[str, bool]:
        """Resolve the target URL and determine if format conversion is needed.

        Args:
            endpoint: The requested endpoint path

        Returns:
            Tuple of (target_url, needs_conversion)
        """
        ...

    @abstractmethod
    async def _create_handler_config(
        self,
        needs_conversion: bool,
        request_context: "RequestContext | None" = None,
    ) -> HandlerConfig:
        """Create handler configuration based on conversion needs.

        Args:
            needs_conversion: Whether format conversion is needed
            request_context: Request context for creating metrics collector

        Returns:
            HandlerConfig instance
        """
        ...

    @abstractmethod
    async def _should_buffer_stream(
        self, request_data: dict[str, Any], is_streaming: bool
    ) -> bool:
        """Determine if a non-streaming request should use buffered streaming internally.

        This method allows plugins to override the default behavior and decide
        when to convert a non-streaming request to a streaming request internally,
        buffer the entire response, and then return it as a non-streaming response.

        Args:
            request_data: Parsed request body data
            is_streaming: Whether the original request is streaming

        Returns:
            True if buffered streaming should be used, False otherwise
        """
        ...

    async def _handle_buffered_streaming(
        self,
        method: str,
        target_url: str,
        body: bytes,
        headers: dict[str, str],
        handler_config: HandlerConfig,
        endpoint: str,
        request_context: "RequestContext",
    ) -> Response:
        """Handle buffered streaming by converting non-streaming request to stream internally.

        This method orchestrates the conversion of a non-streaming request to a streaming
        request internally, buffers the entire response, and converts it back to a
        non-streaming response while maintaining full observability.

        Args:
            method: HTTP method
            target_url: Target API URL
            body: Request body
            headers: Request headers
            handler_config: Handler configuration with SSE parser
            endpoint: Original endpoint for logging
            request_context: Request context for observability

        Returns:
            Non-streaming Response with buffered content

        Raises:
            HTTPException: If buffer service is not available or streaming fails
        """
        # Ensure buffer service is available (lazy initialization if needed)
        buffer_service = await self._get_buffer_service()
        if not buffer_service:
            raise HTTPException(
                status_code=503, detail="Streaming buffer service not available"
            )

        # Extract provider name for hook events
        provider_name = self.__class__.__name__.replace("Adapter", "")

        logger.debug(
            "initiating_buffered_streaming",
            method=method,
            url=target_url,
            endpoint=endpoint,
            provider=provider_name,
            request_id=getattr(request_context, "request_id", None),
        )

        # Delegate to buffer service for stream-to-buffer conversion
        try:
            response = await buffer_service.handle_buffered_streaming_request(
                method=method,
                url=target_url,
                headers=headers,
                body=body,
                handler_config=handler_config,
                request_context=request_context,
                provider_name=provider_name,
            )

            # Apply format adapter if available to convert response format
            logger.debug(
                "checking_format_adapter_conditions",
                has_response_adapter=bool(handler_config.response_adapter),
                response_adapter_type=type(handler_config.response_adapter).__name__
                if handler_config.response_adapter
                else None,
                has_body_attr=hasattr(response, "body"),
                response_type=type(response).__name__,
                request_id=getattr(request_context, "request_id", None),
            )
            if handler_config.response_adapter and hasattr(response, "body"):
                try:
                    import json

                    # Parse the response body
                    body_data = (
                        bytes(response.body)
                        if isinstance(response.body, memoryview)
                        else response.body
                    )
                    response_data = json.loads(body_data)

                    logger.debug(
                        "calling_format_adapter_adapt_response",
                        response_data_type=type(response_data).__name__,
                        response_data_keys=list(response_data.keys())
                        if isinstance(response_data, dict)
                        else None,
                        request_id=getattr(request_context, "request_id", None),
                    )

                    logger.debug(
                        "response_adapter", adapter=handler_config.response_adapter
                    )
                    # Apply format adapter to convert response format (e.g., Codex -> OpenAI)
                    adapted_data = await handler_config.response_adapter.adapt_response(
                        response_data
                    )

                    logger.debug(
                        "format_adapter_adapt_response_completed",
                        adapted_data_type=type(adapted_data).__name__
                        if adapted_data
                        else "NoneType",
                        adapted_data_keys=list(adapted_data.keys())
                        if isinstance(adapted_data, dict)
                        else None,
                        request_id=getattr(request_context, "request_id", None),
                    )

                    # Check if format adapter returned valid data
                    if adapted_data is None:
                        logger.warning(
                            "format_adapter_returned_none",
                            request_id=getattr(request_context, "request_id", None),
                        )
                        # Fall back to original response
                        return response

                    # Create new response with adapted data
                    logger.debug(
                        "creating_adapted_response",
                        adapted_data_type=type(adapted_data).__name__,
                        status_code=response.status_code,
                        request_id=getattr(request_context, "request_id", None),
                    )
                    from starlette.responses import Response

                    adapted_content = json.dumps(adapted_data).encode()

                    # Safely handle response headers
                    try:
                        response_headers = dict(response.headers)
                    except Exception as header_error:
                        logger.warning(
                            "response_headers_conversion_failed",
                            error=str(header_error),
                            request_id=getattr(request_context, "request_id", None),
                        )
                        response_headers = {"Content-Type": "application/json"}

                    return Response(
                        content=adapted_content,
                        status_code=response.status_code,
                        headers=response_headers,
                        media_type="application/json",
                    )
                except Exception as adapter_error:
                    logger.warning(
                        "format_adapter_failed_in_buffered_streaming",
                        error=str(adapter_error),
                        exc_info=adapter_error,
                        request_id=getattr(request_context, "request_id", None),
                    )
                    # Fall back to original response if format adaptation fails

            return response
        except Exception as e:
            logger.error(
                "buffered_streaming_failed",
                method=method,
                url=target_url,
                endpoint=endpoint,
                provider=provider_name,
                error=str(e),
                request_id=getattr(request_context, "request_id", None),
                exc_info=e,
            )
            raise

    async def _get_buffer_service(self) -> "StreamingBufferService | None":
        """Get or create the streaming buffer service.

        Returns:
            StreamingBufferService instance or None if creation fails
        """
        if self._buffer_service is not None:
            return self._buffer_service

        # Lazy initialization - create buffer service with current dependencies
        try:
            # Get HTTP client from pool manager if available for hook-enabled client
            http_client = await self._get_http_client()

            self._buffer_service = StreamingBufferService(
                http_client=http_client,
                request_tracer=self.request_tracer,
                hook_manager=self._hook_manager,
                http_pool_manager=self._http_pool_manager,
            )
            logger.debug("buffer_service_created_lazily")
            return self._buffer_service
        except Exception as e:
            logger.warning(
                "buffer_service_creation_failed",
                error=str(e),
                exc_info=e,
            )
            return None

    async def _execute_request(
        self,
        method: str,
        target_url: str,
        body: bytes,
        auth_headers: dict[str, str],
        access_token: str | None,
        request_headers: dict[str, str],
        handler_config: HandlerConfig,
        endpoint: str,
        needs_conversion: bool,
        request_context: "RequestContext",
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Execute the HTTP request.

        Args:
            method: HTTP method
            target_url: Target API URL
            body: Request body
            auth_headers: Authentication headers
            access_token: Access token if available
            request_headers: Original request headers
            handler_config: Handler configuration
            endpoint: Original endpoint for logging
            needs_conversion: Whether conversion was needed for logging
            request_context: Request context for observability

        Returns:
            Response or StreamingResponse
        """
        # Handler is guaranteed to exist after _validate_prerequisites
        assert self._http_handler is not None

        # Prepare request
        (
            transformed_body,
            headers,
            is_streaming,
        ) = await self._http_handler.prepare_request(
            request_body=body,
            handler_config=handler_config,
            auth_headers=auth_headers,
            request_headers=request_headers,
            access_token=access_token,
        )

        # Parse request body to extract model and other metadata
        try:
            request_data = json.loads(transformed_body) if transformed_body else {}
        except json.JSONDecodeError:
            request_data = {}

        # Update context with provider-specific metadata
        await self._update_request_context(
            request_context, endpoint, request_data, is_streaming, needs_conversion
        )

        # Log the request
        self._log_request(
            endpoint, request_context, is_streaming, needs_conversion, target_url
        )

        # Check if we should buffer stream for non-streaming requests
        if not is_streaming and await self._should_buffer_stream(
            request_data, is_streaming
        ):
            logger.debug(
                "using_buffered_streaming_mode",
                endpoint=endpoint,
                request_id=request_context.request_id,
            )
            return await self._handle_buffered_streaming(
                method=method,
                target_url=target_url,
                body=transformed_body,
                headers=headers,
                handler_config=handler_config,
                endpoint=endpoint,
                request_context=request_context,
            )

        # Get streaming handler if needed
        streaming_handler = self.streaming_handler if is_streaming else None

        # Emit PROVIDER_REQUEST_SENT hook before sending to provider
        hook_manager = self._get_hook_manager()
        if hook_manager:
            provider_name = self.__class__.__name__.replace("Adapter", "")
            try:
                hook_context = HookContext(
                    event=HookEvent.PROVIDER_REQUEST_SENT,
                    timestamp=datetime.now(),
                    provider=provider_name,
                    data={
                        "url": target_url,
                        "method": method,
                        "headers": dict(headers),
                        "is_streaming": is_streaming,
                        "endpoint": endpoint,
                        "model": request_data.get("model", "unknown"),
                        "body": request_data,  # Include full request body for debugging
                    },
                    metadata={
                        "request_id": request_context.request_id,
                        "needs_conversion": needs_conversion,
                    },
                )
                await hook_manager.emit_with_context(hook_context)
            except Exception as e:
                logger.debug(
                    "hook_emission_failed",
                    event="PROVIDER_REQUEST_SENT",
                    error=str(e),
                    category="hooks",
                )

        # Execute request with proper request_context
        try:
            response = await self._http_handler.handle_request(
                method=method,
                url=target_url,
                headers=headers,
                body=transformed_body,
                handler_config=handler_config,
                is_streaming=is_streaming,
                streaming_handler=streaming_handler,
                request_context=request_context,
            )

            # Invalidate credential cache on unauthorized responses to force reload next time
            try:
                if (
                    not is_streaming
                    and hasattr(response, "status_code")
                    and getattr(response, "status_code", 0) == 401
                ):
                    manager: Any = getattr(self, "_auth_manager", None)
                    if manager is not None and hasattr(manager, "clear_cache"):
                        await manager.clear_cache()
            except Exception:
                # Never fail request flow due to cache invalidation issues
                pass

            # Emit PROVIDER_RESPONSE_RECEIVED hook after receiving response
            if hook_manager:
                try:
                    response_data = {
                        "url": target_url,
                        "method": method,
                        "is_streaming": is_streaming,
                        "endpoint": endpoint,
                        "model": request_data.get("model", "unknown"),
                    }

                    # Add response status for non-streaming responses
                    if hasattr(response, "status_code"):
                        response_data["status_code"] = response.status_code

                    # Add response headers
                    if hasattr(response, "headers"):
                        response_data["headers"] = dict(response.headers)

                    # Add response body for debugging (only for non-streaming responses)
                    if not is_streaming and hasattr(response, "text"):
                        try:
                            # Try to get response text
                            response_data["body"] = response.text
                        except Exception:
                            # If text fails, try to get content as bytes
                            try:
                                if hasattr(response, "content"):
                                    response_data["body"] = response.content.decode(
                                        "utf-8", errors="ignore"
                                    )
                            except Exception:
                                response_data["body"] = None

                    response_hook_context = HookContext(
                        event=HookEvent.PROVIDER_RESPONSE_RECEIVED,
                        timestamp=datetime.now(),
                        provider=provider_name,
                        data=response_data,
                        metadata={
                            "request_id": request_context.request_id,
                            "needs_conversion": needs_conversion,
                        },
                        response=response,
                    )
                    await hook_manager.emit_with_context(response_hook_context)
                except Exception as e:
                    logger.debug(
                        "hook_emission_failed",
                        event="PROVIDER_RESPONSE_RECEIVED",
                        error=str(e),
                        category="hooks",
                    )

        except Exception as e:
            # Emit PROVIDER_ERROR hook on error
            if hook_manager:
                try:
                    error_hook_context = HookContext(
                        event=HookEvent.PROVIDER_ERROR,
                        timestamp=datetime.now(),
                        provider=provider_name,
                        data={
                            "url": target_url,
                            "method": method,
                            "endpoint": endpoint,
                            "error": str(e),
                        },
                        metadata={
                            "request_id": request_context.request_id,
                        },
                        error=e,
                    )
                    await hook_manager.emit_with_context(error_hook_context)
                except Exception as hook_error:
                    logger.debug(
                        "hook_emission_failed",
                        event="PROVIDER_ERROR",
                        error=str(hook_error),
                        category="hooks",
                    )
            # Re-raise the original error
            raise

        # Post-process response based on type
        return await self._post_process_response(
            response, is_streaming, request_context
        )

    @abstractmethod
    async def _update_request_context(
        self,
        request_context: "RequestContext",
        endpoint: str,
        request_data: dict[str, Any],
        is_streaming: bool,
        needs_conversion: bool,
    ) -> None:
        """Update request context with provider-specific metadata.

        Args:
            request_context: Request context to update
            endpoint: Target endpoint path
            request_data: Parsed request data
            is_streaming: Whether this is a streaming request
            needs_conversion: Whether format conversion is needed
        """
        ...

    @abstractmethod
    def _log_request(
        self,
        endpoint: str,
        request_context: "RequestContext",
        is_streaming: bool,
        needs_conversion: bool,
        target_url: str,
    ) -> None:
        """Log the request with provider-specific information.

        Args:
            endpoint: Target endpoint path
            request_context: Request context with metadata
            is_streaming: Whether this is a streaming request
            needs_conversion: Whether format conversion is needed
            target_url: Target API URL
        """
        ...

    async def _post_process_response(
        self,
        response: Response | StreamingResponse | DeferredStreaming,
        is_streaming: bool,
        request_context: "RequestContext",
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Post-process response based on type.

        Args:
            response: The response to post-process
            is_streaming: Whether this was a streaming request
            request_context: Request context for observability

        Returns:
            Processed response
        """
        # For non-streaming responses, calculate cost based on usage already extracted in processor
        if not is_streaming and request_context:
            await self._calculate_cost_for_usage(request_context)

        # For deferred streaming responses, return directly (metrics collector already has cost calculation)
        if isinstance(response, DeferredStreaming):
            return response

        # For regular streaming responses, wrap to accumulate chunks and extract headers
        if is_streaming and isinstance(response, StreamingResponse):
            return await self._wrap_streaming_response(response, request_context)

        return response

    @abstractmethod
    async def _calculate_cost_for_usage(
        self, request_context: "RequestContext"
    ) -> None:
        """Calculate cost for usage data already extracted in processor.

        Args:
            request_context: Request context with usage data from processor
        """
        ...

    @abstractmethod
    async def _wrap_streaming_response(
        self, response: StreamingResponse, request_context: "RequestContext"
    ) -> StreamingResponse:
        """Wrap streaming response to accumulate chunks and extract headers.

        Args:
            response: The streaming response to wrap
            request_context: The request context to update

        Returns:
            Wrapped streaming response
        """
        ...

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            # Cleanup HTTP handler
            if self._http_handler and hasattr(self._http_handler, "cleanup"):
                await self._http_handler.cleanup()

            # Clear references
            self._request_transformer = None
            self._response_transformer = None
            # Set observability to null implementations
            self.request_tracer = NullRequestTracer()
            self.metrics = NullMetricsCollector()
            self.streaming_handler = NullStreamingHandler()

            logger.debug("http_adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "http_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )
