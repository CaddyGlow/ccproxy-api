"""Simplified Codex adapter using delegation pattern."""

from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    from ccproxy.adapters.base import APIAdapter
    from ccproxy.services.adapters.format_registry import FormatAdapterRegistry

from fastapi import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.auth.manager import AuthManager
from ccproxy.config.utils import (
    ANTHROPIC_MESSAGES_PATH,
    CODEX_API_BASE_URL,
    CODEX_RESPONSES_ENDPOINT,
    OPENAI_CHAT_COMPLETIONS_PATH,
    OPENAI_COMPLETIONS_PATH,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.http.streaming.sse import last_json_data_event
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.services.handler_config import HandlerConfig


if TYPE_CHECKING:
    from ccproxy.core.plugins import PluginContext
    from ccproxy.core.plugins.hooks import HookManager
    from ccproxy.core.request_context import RequestContext
    from ccproxy.services.interfaces import (
        IMetricsCollector,
        IRequestTracer,
        StreamingMetrics,
    )

    from .detection_service import CodexDetectionService

from ccproxy.services.adapters.format_context import FormatContext

from .models import CodexAuthData
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
        auth_manager: AuthManager,
        detection_service: "CodexDetectionService",
        http_pool_manager: Any,
        # Optional dependencies
        request_tracer: "IRequestTracer | None" = None,
        metrics: "IMetricsCollector | None" = None,
        streaming_handler: "StreamingMetrics | None" = None,
        hook_manager: "HookManager | None" = None,
        # Format services
        format_registry: "FormatAdapterRegistry | None" = None,
        # Plugin-specific context
        context: "PluginContext | dict[str, Any] | None" = None,
    ):
        """Initialize the Codex adapter with explicit dependencies.

        Args:
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for Codex CLI detection
            http_pool_manager: HTTP pool manager for getting clients on demand
            request_tracer: Optional request tracer
            metrics: Optional metrics collector
            streaming_handler: Optional streaming handler
            hook_manager: Optional hook manager for event emission
            format_registry: Format adapter registry for protocol conversions
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

        # Assign format services from constructor parameters
        self.format_registry = format_registry

        logger.debug(
            "format_services_loaded",
            has_registry=bool(self.format_registry),
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
        # Store current endpoint for format adapter selection
        self._current_endpoint = endpoint

        # Check if format conversion is needed based on endpoint
        # OpenAI and Anthropic format endpoints need conversion to Codex format
        needs_conversion = (
            endpoint.endswith(OPENAI_CHAT_COMPLETIONS_PATH)
            or endpoint.endswith(OPENAI_COMPLETIONS_PATH)
            or endpoint.endswith(ANTHROPIC_MESSAGES_PATH)
        )

        logger.info(
            "codex_resolve_endpoint",
            endpoint=endpoint,
            needs_conversion=needs_conversion,
            current_endpoint=self._current_endpoint,
        )

        # Build target URL (always uses Codex responses endpoint)
        target_url = f"{CODEX_API_BASE_URL}{CODEX_RESPONSES_ENDPOINT}"

        return target_url, needs_conversion

    def _get_format_adapter(self, source_format: str) -> "APIAdapter":
        """Get format adapter with fail-fast error handling.

        Args:
            source_format: The source format to convert from

        Returns:
            Format adapter for the conversion
        """
        if self.format_registry is None:
            raise RuntimeError("Format registry is not available")
        try:
            return cast(
                "APIAdapter",
                self.format_registry.get_adapter(source_format, "response_api"),
            )
        except ValueError as e:
            logger.error(
                "format_adapter_not_found", source_format=source_format, error=str(e)
            )
            raise RuntimeError(
                f"No format adapter found for {source_format}->response_api"
            ) from e

    def _get_format_from_endpoint(self, endpoint: str) -> str:
        """Get source format from endpoint using constants.

        Args:
            endpoint: The endpoint path

        Returns:
            Source format string

        Raises:
            ValueError: If endpoint format cannot be determined
        """
        if endpoint.endswith(OPENAI_CHAT_COMPLETIONS_PATH):
            return "openai"
        elif endpoint.endswith(ANTHROPIC_MESSAGES_PATH):
            return "anthropic"
        elif CODEX_RESPONSES_ENDPOINT in endpoint:
            return "response_api"
        else:
            raise ValueError(f"Unable to detect format from endpoint: {endpoint}")

    async def _extract_provider_auth_data(self) -> CodexAuthData:
        """Extract provider-specific authentication data including access token.

        For Codex/OpenAI, extract access_token and chatgpt_account_id from user profile.

        Returns:
            Typed authentication data for Codex/OpenAI provider
        """
        # Get access token
        access_token = await self._auth_manager.get_access_token()

        # Get chatgpt_account_id
        chatgpt_account_id = None
        try:
            user_profile = await self._auth_manager.get_user_profile()
            logger.debug(
                "auth_profile_extraction",
                has_profile=user_profile is not None,
                profile_type=type(user_profile).__name__ if user_profile else None,
                has_chatgpt_account_id_attr=hasattr(user_profile, "chatgpt_account_id")
                if user_profile
                else False,
            )

            if user_profile and hasattr(user_profile, "chatgpt_account_id"):
                account_id = getattr(user_profile, "chatgpt_account_id", None)
                logger.debug(
                    "chatgpt_account_id_extraction",
                    raw_account_id=account_id,
                    account_id_type=type(account_id).__name__
                    if account_id is not None
                    else None,
                    is_string=isinstance(account_id, str),
                    is_truthy=bool(account_id),
                )
                if account_id and isinstance(account_id, str):
                    chatgpt_account_id = account_id
                    logger.debug(
                        "chatgpt_account_id_set",
                        account_id_length=len(account_id),
                    )

            # Also debug the extras content to see JWT claims structure
            if user_profile and hasattr(user_profile, "extras"):
                extras = getattr(user_profile, "extras", {})
                auth_claims = extras.get("https://api.openai.com/auth", {})
                logger.debug(
                    "jwt_claims_debug",
                    has_extras=bool(extras),
                    extras_keys=list(extras.keys())
                    if isinstance(extras, dict)
                    else None,
                    has_auth_claims=bool(auth_claims),
                    auth_claims_keys=list(auth_claims.keys())
                    if isinstance(auth_claims, dict)
                    else None,
                    raw_chatgpt_account_id=auth_claims.get("chatgpt_account_id")
                    if isinstance(auth_claims, dict)
                    else None,
                )

        except Exception as e:
            logger.warning(
                "chatgpt_account_id_extraction_failed",
                error=str(e),
                error_type=type(e).__name__,
            )

        return CodexAuthData(
            access_token=access_token,
            chatgpt_account_id=chatgpt_account_id,
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
        format_adapter = None
        format_context = None

        if needs_conversion:
            try:
                if self._current_endpoint is None:
                    raise RuntimeError("Current endpoint is not set")
                source_format = self._get_format_from_endpoint(self._current_endpoint)
                format_adapter = self._get_format_adapter(source_format)
                format_context = FormatContext(
                    source_format=source_format,
                    target_format="response_api",
                    conversion_needed=True,
                    streaming_mode="streaming",  # Use config.preferred_upstream_mode from context
                )
            except (ValueError, RuntimeError) as e:
                logger.error(
                    "format_detection_failed",
                    endpoint=self._current_endpoint,
                    error=str(e),
                )
                raise RuntimeError(
                    f"Format detection failed for endpoint {self._current_endpoint}"
                ) from e

        # Provide an SSE parser that satisfies the SSEParserProtocol
        class _SSEParser:
            def __call__(self, raw: str) -> dict[str, Any] | None:
                return last_json_data_event(raw)

            def transform_body(self, body: Any) -> Any:
                return body

        return HandlerConfig(
            request_adapter=format_adapter,
            response_adapter=format_adapter,
            format_context=format_context,
            request_transformer=self._request_transformer,
            response_transformer=self._response_transformer,
            supports_streaming=True,
            sse_parser=_SSEParser(),
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
        session_id = kwargs.get("session_id")

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
        from ccproxy.plugins.pricing.exceptions import (
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
        """Handle a streaming request to the Codex API."""
        # Delegate to the base handler - buffer service will handle non-streaming conversion
        result = await self.handle_request(request, endpoint, "POST", **kwargs)

        # Ensure we return a streaming response
        if not isinstance(result, StreamingResponse):
            return StreamingResponse(
                iter([result.body if hasattr(result, "body") else b""]),
                media_type="text/event-stream",
            )

        return result

    async def _should_buffer_stream(
        self, request_data: dict[str, Any], is_streaming: bool
    ) -> bool:
        """Use configuration to determine buffering with fail-fast validation.

        Args:
            request_data: Parsed request body data
            is_streaming: Whether the client requested streaming

        Returns:
            True if should convert to buffered streaming, False otherwise
        """
        # Get config from context - this should be available from plugin initialization
        config = None
        if hasattr(self, "_context") and self._context:
            config = self._context.get("config")

        if not config:
            # Fallback to default behavior if config not available
            logger.warning(
                "codex_config_not_available", using_default_buffer_behavior=True
            )
            return not is_streaming

        # Respect configuration settings
        if not getattr(config, "buffer_non_streaming", True):
            return False

        # If preferred mode is streaming and request is not streaming, buffer it
        preferred_mode = getattr(config, "preferred_upstream_mode", "streaming")
        return preferred_mode == "streaming" and not is_streaming

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
