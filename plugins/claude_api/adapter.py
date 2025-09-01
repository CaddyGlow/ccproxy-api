"""Claude API adapter implementation."""

import json
from typing import TYPE_CHECKING, Any, cast

from fastapi import HTTPException, Request
from httpx import AsyncClient
from starlette.responses import StreamingResponse


if TYPE_CHECKING:
    from ccproxy.auth.manager import AuthManager
    from ccproxy.core.request_context import RequestContext
    from ccproxy.hooks import HookManager
    from ccproxy.plugins.declaration import PluginContext
    from ccproxy.services.interfaces import (
        IMetricsCollector,
        IRequestTracer,
        IStreamingHandler,
    )
    from plugins.claude_api.detection_service import ClaudeAPIDetectionService

from ccproxy.config.constants import (
    CLAUDE_API_BASE_URL,
    CLAUDE_MESSAGES_ENDPOINT,
    OPENAI_CHAT_COMPLETIONS_PATH,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.streaming.deferred_streaming import DeferredStreaming

from .transformers import ClaudeAPIRequestTransformer, ClaudeAPIResponseTransformer


logger = get_plugin_logger()


class ClaudeAPIAdapter(BaseHTTPAdapter):
    """Claude API adapter implementation.

    This adapter provides direct access to the Anthropic Claude API
    with support for both native Anthropic format and OpenAI-compatible format.
    """

    def __init__(
        self,
        # Required dependencies
        http_client: AsyncClient,
        auth_manager: "AuthManager",
        detection_service: "ClaudeAPIDetectionService",
        # Optional dependencies
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "IStreamingHandler | None" = None,
        hook_manager: "HookManager | None" = None,
        # Plugin-specific context
        context: "PluginContext | dict[str, Any] | None" = None,
    ) -> None:
        """Initialize the Claude API adapter with explicit dependencies.

        Args:
            http_client: HTTP client for making requests
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for Claude CLI detection
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            hook_manager: Optional hook manager for event emission
            context: Optional plugin context containing plugin_registry and other services
        """
        # Get injection mode from config if available
        injection_mode = "minimal"  # default
        if context:
            # Handle both dict and TypedDict formats
            if isinstance(context, dict):
                config = context.get("config")
            else:
                config = getattr(context, "config", None)

            if config:
                injection_mode = getattr(
                    config, "system_prompt_injection_mode", "minimal"
                )

        # Initialize transformers with injection mode
        request_transformer = ClaudeAPIRequestTransformer(
            detection_service, mode=injection_mode
        )

        # Get CORS settings if available
        cors_settings = None
        # Try from context if available
        if context:
            config = (
                context.get("config")
                if isinstance(context, dict)
                else getattr(context, "config", None)
            )
            if config:
                cors_settings = getattr(config, "cors", None)

        response_transformer = ClaudeAPIResponseTransformer(cors_settings)

        # Initialize base HTTP adapter with explicit dependencies
        super().__init__(
            http_client=http_client,
            auth_manager=auth_manager,
            detection_service=detection_service,
            request_tracer=request_tracer,
            metrics=metrics,
            streaming_handler=streaming_handler,
            request_transformer=request_transformer,
            response_transformer=response_transformer,
            hook_manager=hook_manager,
            context=cast("PluginContext | None", context),
        )

        # Initialize OpenAI adapter for format conversion
        from ccproxy.adapters.openai.adapter import OpenAIAdapter

        self.openai_adapter: OpenAIAdapter | None = OpenAIAdapter()

    async def _resolve_endpoint(self, endpoint: str) -> tuple[str, bool]:
        """Resolve the target URL and determine if format conversion is needed.

        Args:
            endpoint: The requested endpoint path

        Returns:
            Tuple of (target_url, needs_conversion)
        """
        if endpoint.endswith(CLAUDE_MESSAGES_ENDPOINT):
            # Native Anthropic format
            return f"{CLAUDE_API_BASE_URL}{CLAUDE_MESSAGES_ENDPOINT}", False
        elif endpoint.endswith(OPENAI_CHAT_COMPLETIONS_PATH):
            # OpenAI format - needs conversion
            return f"{CLAUDE_API_BASE_URL}{CLAUDE_MESSAGES_ENDPOINT}", True
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Endpoint {endpoint} not supported by Claude API plugin",
            )

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
        return HandlerConfig(
            request_adapter=self.openai_adapter if needs_conversion else None,
            response_adapter=self.openai_adapter if needs_conversion else None,
            request_transformer=self._request_transformer,
            response_transformer=self._response_transformer,
            supports_streaming=True,
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
                "provider": "claude_api",
                "service_type": "claude_api",
                "endpoint": endpoint.rstrip("/").split("/")[-1]
                if endpoint
                else "messages",
                "model": request_data.get("model", "unknown"),
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
            plugin="claude_api",
            endpoint=endpoint,
            model=request_context.metadata.get("model"),
            is_streaming=is_streaming,
            needs_conversion=needs_conversion,
            target_url=target_url,
        )

    def _get_pricing_service(self) -> Any | None:
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

    async def _calculate_cost_for_usage(
        self, request_context: "RequestContext"
    ) -> None:
        """Calculate cost for usage data already extracted in processor.

        Args:
            request_context: Request context with usage data from processor
        """
        # Check if we have usage data from the processor
        metadata = request_context.metadata
        tokens_input = metadata.get("tokens_input", 0)
        tokens_output = metadata.get("tokens_output", 0)

        # Skip if no usage data available
        if not (tokens_input or tokens_output):
            return

        # Get pricing service and calculate cost
        pricing_service = self._get_pricing_service()
        if not pricing_service:
            return

        try:
            model = metadata.get("model", "claude-3-5-sonnet-20241022")
            cache_read_tokens = metadata.get("cache_read_tokens", 0)
            cache_write_tokens = metadata.get("cache_write_tokens", 0)

            # Import pricing exceptions
            from plugins.pricing.exceptions import (
                ModelPricingNotFoundError,
                PricingDataNotLoadedError,
                PricingServiceDisabledError,
            )

            cost_decimal = await pricing_service.calculate_cost(
                model_name=model,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
            )
            cost_usd = float(cost_decimal)

            # Update context with calculated cost
            metadata["cost_usd"] = cost_usd

            logger.debug(
                "cost_calculated",
                model=model,
                cost_usd=cost_usd,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                source="non_streaming",
            )
        except ModelPricingNotFoundError as e:
            logger.warning(
                "model_pricing_not_found",
                model=model,
                message=str(e),
                tokens_input=tokens_input,
                tokens_output=tokens_output,
            )
        except PricingDataNotLoadedError as e:
            logger.warning(
                "pricing_data_not_loaded",
                model=model,
                message=str(e),
            )
        except PricingServiceDisabledError as e:
            logger.debug(
                "pricing_service_disabled",
                message=str(e),
            )
        except Exception as e:
            logger.debug(
                "cost_calculation_failed",
                error=str(e),
                model=metadata.get("model"),
            )

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
        from collections.abc import AsyncIterator

        # Get the original iterator
        original_iterator = response.body_iterator

        # Create accumulator for chunks
        chunks: list[bytes] = []
        headers_extracted = False

        # Note: Metrics extraction is now handled by ClaudeAPIStreamingMetricsHook

        async def wrapped_iterator() -> AsyncIterator[bytes]:
            """Wrap the stream iterator to accumulate chunks."""
            nonlocal headers_extracted

            async for chunk in original_iterator:
                # Extract headers on first chunk (after streaming has started)
                if not headers_extracted:
                    headers_extracted = True
                    if "response_headers" in request_context.metadata:
                        response_headers = request_context.metadata["response_headers"]

                        # Extract relevant headers and put them directly in metadata for access_logger
                        headers_for_log = {}
                        for k, v in response_headers.items():
                            k_lower = k.lower()
                            # Include Anthropic headers and request IDs
                            if k_lower.startswith("anthropic-ratelimit"):
                                # Put rate limit headers directly in metadata for access_logger
                                request_context.metadata[k_lower] = v
                                headers_for_log[k] = v
                            elif k_lower == "anthropic-request-id":
                                # Also store request ID
                                request_context.metadata["anthropic_request_id"] = v
                                headers_for_log[k] = v
                            elif "request" in k_lower and "id" in k_lower:
                                headers_for_log[k] = v

                        # Also store the headers dictionary for display
                        request_context.metadata["headers"] = headers_for_log

                        logger.debug(
                            "claude_api_headers_extracted",
                            headers_count=len(headers_for_log),
                            headers=headers_for_log,
                            direct_metadata_keys=[
                                k
                                for k in request_context.metadata
                                if "anthropic" in k.lower()
                            ],
                            category="http",
                        )

                if isinstance(chunk, str | memoryview):
                    chunk = chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                chunks.append(chunk)

                # Note: Chunk processing for metrics is handled by hooks

                yield chunk

            # Mark that stream processing is complete
            request_context.metadata.update(
                {
                    "stream_accumulated": True,
                    "stream_chunks_count": len(chunks),
                }
            )

        # Create new streaming response with wrapped iterator
        return StreamingResponse(
            wrapped_iterator(),
            status_code=response.status_code,
            headers=dict(response.headers) if hasattr(response, "headers") else {},
            media_type=response.media_type,
        )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse | DeferredStreaming:
        """Handle a streaming request to the Claude API.

        Forces stream=true in the request body and delegates to handle_request.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            **kwargs: Additional arguments

        Returns:
            Streaming response from Claude API
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

            # Claude API specific cleanup
            self.openai_adapter = None

            logger.debug("claude_api_adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "claude_api_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )
