"""Simplified Codex adapter using delegation pattern."""

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any, cast

from fastapi import Request
from httpx import AsyncClient
from starlette.responses import Response, StreamingResponse

from ccproxy.auth.manager import AuthManager
from ccproxy.config.constants import (
    CODEX_API_BASE_URL,
    CODEX_RESPONSES_ENDPOINT,
    OPENAI_CHAT_COMPLETIONS_PATH,
    OPENAI_COMPLETIONS_PATH,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.handler_config import HandlerConfig


if TYPE_CHECKING:
    from ccproxy.core.request_context import RequestContext
    from ccproxy.hooks import HookManager
    from ccproxy.plugins.declaration import PluginContext
    from ccproxy.services.interfaces import (
        IMetricsCollector,
        IRequestTracer,
        IStreamingHandler,
    )
    from plugins.codex.detection_service import CodexDetectionService

from .format_adapter import CodexFormatAdapter
from .transformers import CodexRequestTransformer, CodexResponseTransformer


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Codex adapter using HTTP adapter delegation pattern.

    This adapter follows the same pattern as Claude API adapter,
    delegating actual HTTP operations to base HTTP adapter.
    """

    def __init__(
        self,
        # Required dependencies
        http_client: AsyncClient,
        auth_manager: AuthManager,
        detection_service: "CodexDetectionService",
        # Optional dependencies
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "IStreamingHandler | None" = None,
        hook_manager: "HookManager | None" = None,
        # Plugin-specific context
        context: "PluginContext | dict[str, Any] | None" = None,
    ):
        """Initialize the Codex adapter with explicit dependencies.

        Args:
            http_client: HTTP client for making requests
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for Codex CLI detection
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            hook_manager: Optional hook manager for event emission
            context: Optional plugin context containing plugin_registry and other services
        """
        # Initialize transformers
        request_transformer = CodexRequestTransformer(detection_service)

        # Initialize response transformer with CORS settings
        cors_settings = None
        # Try from context
        if context:
            config = (
                context.get("config")
                if isinstance(context, dict)
                else getattr(context, "config", None)
            )
            if config:
                cors_settings = getattr(config, "cors", None)

        response_transformer = CodexResponseTransformer(cors_settings)

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

        # Initialize components
        self.format_adapter = CodexFormatAdapter()

        # Complete initialization if needed
        self._complete_initialization()

    async def _resolve_endpoint(self, endpoint: str) -> tuple[str, bool]:
        """Resolve the target URL and determine if format conversion is needed.

        Args:
            endpoint: The requested endpoint path

        Returns:
            Tuple of (target_url, needs_conversion)
        """
        # Check if format conversion is needed based on endpoint
        # OpenAI format endpoints need conversion to Codex format
        needs_conversion = endpoint.endswith(
            OPENAI_CHAT_COMPLETIONS_PATH
        ) or endpoint.endswith(OPENAI_COMPLETIONS_PATH)

        # Build target URL (always uses Codex responses endpoint)
        target_url = f"{CODEX_API_BASE_URL}{CODEX_RESPONSES_ENDPOINT}"

        return target_url, needs_conversion

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
            request_adapter=self.format_adapter if needs_conversion else None,
            response_adapter=self.format_adapter if needs_conversion else None,
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
                "provider": "codex",
                "service_type": "codex",
                "endpoint": endpoint.rstrip("/").split("/")[-1]
                if endpoint
                else "responses",
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
        logger.debug(
            "plugin_request",
            plugin="codex",
            endpoint=endpoint,
            model=request_context.metadata.get("model"),
            is_streaming=is_streaming,
            needs_conversion=needs_conversion,
            target_url=target_url,
        )

    def _complete_initialization(self) -> None:
        """Complete initialization - no longer needed with explicit dependencies."""
        # HTTP handler is already initialized by BaseHTTPAdapter
        # Transformers are already set from constructor
        pass

    async def handle_request(
        self, request: Request, endpoint: str, method: str, **kwargs: Any
    ) -> Response | StreamingResponse:
        """Handle a request to the Codex API.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            method: HTTP method
            **kwargs: Additional arguments (e.g., session_id)

        Returns:
            Response from Codex API
        """
        # Extract session_id for Codex-specific handling
        session_id = kwargs.get("session_id") or str(uuid.uuid4())

        # Store session_id in request state for prepare_request
        request.state.codex_session_id = session_id

        # Delegate to parent handle_request implementation
        return await super().handle_request(request, endpoint, method, **kwargs)

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

        # Import pricing exceptions
        from plugins.pricing.exceptions import (
            ModelPricingNotFoundError,
            PricingDataNotLoadedError,
            PricingServiceDisabledError,
        )

        try:
            model = metadata.get("model", "gpt-3.5-turbo")
            cache_read_tokens = metadata.get("cache_read_tokens", 0)

            cost_decimal = await pricing_service.calculate_cost(
                model_name=model,
                input_tokens=tokens_input,
                output_tokens=tokens_output,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=0,  # OpenAI doesn't have cache write tokens
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
                model=model,
            )

    async def handle_streaming(
        self, request: Request, endpoint: str, **kwargs: Any
    ) -> StreamingResponse:
        """Handle a streaming request to the Codex API.

        Args:
            request: FastAPI request object
            endpoint: Target endpoint path
            **kwargs: Additional arguments

        Returns:
            Streaming response from Codex API
        """

        # Ensure stream=true in request body
        body = await request.body()
        request_data = {}
        if body:
            with contextlib.suppress(json.JSONDecodeError):
                request_data = json.loads(body)

        # Force streaming
        request_data["stream"] = True
        modified_body = json.dumps(request_data).encode()

        # Create modified request with stream=true
        modified_scope = {
            **request.scope,
            "_body": modified_body,
        }

        from starlette.requests import Request as StarletteRequest

        modified_request = StarletteRequest(
            scope=modified_scope,
            receive=request.receive,
        )
        modified_request._body = modified_body

        # Delegate to handle_request which will handle streaming
        result = await self.handle_request(modified_request, endpoint, "POST", **kwargs)

        # Ensure we return a streaming response
        if not isinstance(result, StreamingResponse):
            return StreamingResponse(
                iter([result.body if hasattr(result, "body") else b""]),
                media_type="text/event-stream",
            )

        return result

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            # Call parent cleanup first
            await super().cleanup()

            # Codex-specific cleanup
            # No additional references to clear

            logger.debug("codex_adapter_cleanup_completed")

        except Exception as e:
            logger.error(
                "codex_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )

    async def _wrap_streaming_response(
        self, response: StreamingResponse, request_context: "RequestContext"
    ) -> StreamingResponse:
        """Wrap streaming response to accumulate chunks and extract headers.

        For Codex/OpenAI, we pass through the streaming response as-is since
        metrics extraction is handled by hooks.

        Args:
            response: The streaming response to wrap
            request_context: The request context to update

        Returns:
            The original streaming response (pass-through)
        """
        # For Codex/OpenAI, metrics extraction is handled by hooks
        # so we can just pass through the response
        return response
