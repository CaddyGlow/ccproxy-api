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
from ccproxy.streaming.deferred_streaming import DeferredStreaming


if TYPE_CHECKING:
    import httpx

    from ccproxy.auth.manager import AuthManager
    from ccproxy.core.transformers import BaseTransformer
    from ccproxy.observability.context import RequestContext
    from ccproxy.plugins.declaration import PluginContext
    from ccproxy.services.cli_detection import CLIDetectionService
    from ccproxy.streaming.interfaces import IStreamingMetricsCollector


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
        http_client: "httpx.AsyncClient",
        auth_manager: "AuthManager",
        detection_service: "CLIDetectionService",
        # Optional dependencies with defaults
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "IStreamingHandler | None" = None,
        request_transformer: "BaseTransformer | None" = None,
        response_transformer: "BaseTransformer | None" = None,
        # Context for plugin-specific services
        context: "PluginContext | None" = None,
    ) -> None:
        """Initialize the base HTTP adapter with explicit dependencies.

        Args:
            http_client: HTTP client for making requests
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for CLI detection
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            request_transformer: Optional request transformer
            response_transformer: Optional response transformer
            context: Optional plugin context containing plugin_registry and other services
        """
        # Store required dependencies
        self.http_client = http_client
        self._auth_manager = auth_manager
        self._detection_service = detection_service

        # Use null object pattern for optional dependencies
        self.request_tracer = request_tracer or NullRequestTracer()
        self.metrics = metrics or NullMetricsCollector()
        self.streaming_handler = streaming_handler or NullStreamingHandler()

        # Store context for plugin-specific needs
        self.context = context or {}

        # Initialize transformers (can be overridden by subclasses)
        self._request_transformer = request_transformer
        self._response_transformer = response_transformer

        # Initialize HTTP handler with explicit dependencies
        self._http_handler: PluginHTTPHandler = PluginHTTPHandler(
            http_client=http_client, request_tracer=self.request_tracer
        )

    def _get_hook_manager(self) -> "HookManager | None":
        """Get hook manager from context if available."""
        try:
            if not self.context:
                return None

            # Try to get from context directly
            if "hook_manager" in self.context:
                return self.context["hook_manager"]

            # Try to get from app state
            if "app" in self.context and hasattr(
                self.context["app"].state, "hook_manager"
            ):
                return self.context["app"].state.hook_manager

            return None
        except Exception:
            return None

    def _get_pricing_service(self) -> "PricingService | None":
        """Get pricing service from plugin registry if available."""
        try:
            if not self.context or "plugin_registry" not in self.context:
                return None

            plugin_registry = self.context["plugin_registry"]

            # Import locally to avoid circular dependency
            from plugins.pricing.service import PricingService

            # Get service from registry with type checking
            return plugin_registry.get_service("pricing", PricingService)

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

        # Get RequestContext - it must exist when called via ProxyService
        from ccproxy.observability.context import RequestContext

        request_context: RequestContext | None = RequestContext.get_current()
        if not request_context:
            raise HTTPException(
                status_code=500,
                detail="RequestContext not available - plugin must be called via ProxyService",
            )

        # Get request body and auth
        body = await request.body()

        # Get access token directly from auth manager
        access_token = await self._auth_manager.get_access_token()

        # Build auth headers with Bearer token
        auth_headers = {"Authorization": f"Bearer {access_token}"}

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
    async def _create_metrics_collector(
        self, request_context: "RequestContext"
    ) -> "IStreamingMetricsCollector | None":
        """Create a metrics collector for this request.

        Args:
            request_context: Request context containing request_id

        Returns:
            Metrics collector or None
        """
        ...

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
            self.http_client = None  # Clear HTTP client reference
            self.request_tracer = None
            self.metrics = None
            self.streaming_handler = None

            logger.debug("http_adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "http_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )
