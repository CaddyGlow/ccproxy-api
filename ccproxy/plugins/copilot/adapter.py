"""Copilot adapter implementation using BaseHTTPAdapter delegation pattern."""

import json
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request
from starlette.responses import StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming import DeferredStreaming

from .config import CopilotConfig
from .detection_service import CopilotDetectionService
from .format_adapter import CopilotFormatAdapter
from .oauth.provider import CopilotOAuthProvider
from .transformers.request import CopilotRequestTransformer
from .transformers.response import CopilotResponseTransformer


if TYPE_CHECKING:
    from ccproxy.core.plugins import PluginContext
    from ccproxy.core.plugins.hooks import HookManager
    from ccproxy.core.request_context import RequestContext
    from ccproxy.services.interfaces import (
        IMetricsCollector,
        IRequestTracer,
        StreamingMetrics,
    )


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """GitHub Copilot adapter implementation using BaseHTTPAdapter delegation pattern."""

    def __init__(
        self,
        # Required dependencies following BaseHTTPAdapter pattern
        auth_manager: Any,  # AuthManager from ccproxy.auth.manager
        detection_service: CopilotDetectionService,
        http_pool_manager: Any,
        # Copilot-specific dependencies
        oauth_provider: CopilotOAuthProvider,
        config: CopilotConfig,
        # Optional dependencies
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "StreamingMetrics | None" = None,
        hook_manager: "HookManager | None" = None,
        # Plugin-specific context
        context: "PluginContext | dict[str, Any] | None" = None,
    ) -> None:
        """Initialize the Copilot adapter with explicit dependencies.

        Args:
            auth_manager: Authentication manager for credentials (required)
            detection_service: GitHub CLI detection service (required)
            http_pool_manager: HTTP pool manager for getting clients on demand
            oauth_provider: OAuth provider for Copilot authentication (required)
            config: Plugin configuration
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            hook_manager: Optional hook manager for event emission
            context: Optional plugin context containing plugin_registry and other services
        """
        # Store Copilot-specific dependencies
        self.config = config
        self.oauth_provider = oauth_provider
        self.format_adapter = CopilotFormatAdapter()

        # Initialize transformers
        request_transformer = CopilotRequestTransformer(config)
        response_transformer = CopilotResponseTransformer()

        # Initialize base HTTP adapter with explicit dependencies
        super().__init__(
            auth_manager=auth_manager,
            detection_service=detection_service,
            http_pool_manager=http_pool_manager,
            request_tracer=request_tracer,
            metrics=metrics,
            streaming_handler=streaming_handler,
            request_transformer=request_transformer,
            response_transformer=response_transformer,
            hook_manager=hook_manager,
            context=cast("PluginContext | None", context),
        )

        # Current endpoint tracking for format detection
        self._current_endpoint: str | None = None

    async def _resolve_endpoint(self, endpoint: str) -> tuple[str, bool]:
        """Resolve the target URL and determine if format conversion is needed.

        Args:
            endpoint: The requested endpoint path

        Returns:
            Tuple of (target_url, needs_conversion)
        """
        # Store current endpoint for format detection
        self._current_endpoint = endpoint

        # Check if format conversion is needed
        needs_conversion = self._needs_format_conversion(endpoint, {})

        # Build target URL - Copilot API endpoint
        # This should be derived from the endpoint or config
        target_url = (
            "https://api.githubcopilot.com/chat/completions"  # Default Copilot endpoint
        )

        return target_url, needs_conversion

    async def _extract_provider_auth_data(self) -> dict[str, Any]:
        """Extract provider-specific authentication data including access token.

        Returns:
            Dictionary containing auth data including access_token
        """
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
        request_context: "RequestContext | None" = None,
    ) -> HandlerConfig:
        """Create handler configuration based on conversion needs.

        Args:
            needs_conversion: Whether format conversion is needed
            request_context: Request context for creating metrics collector

        Returns:
            HandlerConfig instance
        """
        # For now, don't use format adapters since CopilotFormatAdapter
        # may not implement the APIAdapter interface properly
        # TODO: Update CopilotFormatAdapter to implement APIAdapter interface
        request_adapter = None
        response_adapter = None

        return HandlerConfig(
            request_adapter=request_adapter,
            response_adapter=response_adapter,
            request_transformer=self._request_transformer,
            response_transformer=self._response_transformer,
            supports_streaming=True,
            preserve_header_case=True,
        )

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
        request_context.metadata.update(
            {
                "provider": "copilot",
                "service_type": "copilot",
                "endpoint": endpoint.rstrip("/").split("/")[-1]
                if endpoint
                else "completions",
                "model": request_data.get("model", "gpt-4"),
                "stream": is_streaming,
                "needs_conversion": needs_conversion,
            }
        )

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
        logger.info(
            "plugin_request",
            plugin="copilot",
            endpoint=endpoint,
            model=request_context.metadata.get("model"),
            is_streaming=is_streaming,
            needs_conversion=needs_conversion,
            target_url=target_url,
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
            "/chat/completions",
            "/completions",
            "/embeddings",
            "/models",
        ]

        for openai_endpoint in openai_endpoints:
            if endpoint.endswith(openai_endpoint):
                return True

        return False

    async def _calculate_cost_for_usage(
        self, request_context: "RequestContext"
    ) -> None:
        """Calculate cost for usage data already extracted in processor.

        Args:
            request_context: Request context with usage data from processor
        """
        # TODO: Implement pricing calculation when pricing service is integrated
        logger.debug("copilot_cost_calculation_skipped")

    async def _wrap_streaming_response(
        self, response: StreamingResponse, request_context: "RequestContext"
    ) -> StreamingResponse:
        """Wrap streaming response for metrics and logging.

        Args:
            response: The streaming response to wrap
            request_context: The request context to update

        Returns:
            Wrapped streaming response
        """
        # For now, return response as-is
        # TODO: Add metrics extraction and chunk accumulation
        return response

    async def _should_buffer_stream(
        self, request_data: dict[str, Any], is_streaming: bool
    ) -> bool:
        """Determine if stream should be buffered.

        Args:
            request_data: Parsed request body data
            is_streaming: Whether the original request is streaming

        Returns:
            False (Copilot doesn't use buffered streaming by default)
        """
        # Don't buffer Copilot streams by default
        return False

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request to the Copilot API.

        Forces stream=true in the request body and delegates to handle_request.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            **kwargs: Additional arguments

        Returns:
            Streaming response from Copilot API
        """
        # Modify request to force streaming
        modified_request = await self._create_streaming_request(request)

        # Delegate to handle_request
        result = await self.handle_request(modified_request, endpoint, "POST", **kwargs)

        # Return deferred or streaming response directly
        if isinstance(result, StreamingResponse | DeferredStreaming):
            return result

        # Fallback: wrap non-streaming response
        return StreamingResponse(
            iter([result.body if hasattr(result, "body") else b""]),
            media_type="text/event-stream",
        )

    async def _create_streaming_request(self, request: Request) -> Request:
        """Create a modified request with stream=true.

        Args:
            request: Original request

        Returns:
            Modified request with stream=true
        """
        body = await request.body()

        # Parse and modify request data
        try:
            request_data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            request_data = {}

        request_data["stream"] = True
        modified_body = json.dumps(request_data).encode()

        # Create modified request
        from starlette.requests import Request as StarletteRequest

        modified_scope = {**request.scope, "_body": modified_body}
        modified_request = StarletteRequest(
            scope=modified_scope,
            receive=request.receive,
        )
        modified_request._body = modified_body

        return modified_request

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            # Call parent cleanup first
            await super().cleanup()

            # Copilot-specific cleanup
            if self.oauth_provider:
                await self.oauth_provider.cleanup()

            # Clear references
            self._current_endpoint = None

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
