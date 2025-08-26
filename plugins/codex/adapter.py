"""Simplified Codex adapter using delegation pattern."""

import contextlib
import json
import uuid
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException, Request
from starlette.responses import Response, StreamingResponse

from ccproxy.auth.manager import AuthManager
from ccproxy.config.constants import (
    CODEX_API_BASE_URL,
    CODEX_RESPONSES_ENDPOINT,
    OPENAI_CHAT_COMPLETIONS_PATH,
    OPENAI_COMPLETIONS_PATH,
)
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.base import BaseAdapter
from ccproxy.services.handler_config import HandlerConfig
from ccproxy.services.http.plugin_handler import PluginHTTPHandler
from ccproxy.services.interfaces import IRequestHandler


if TYPE_CHECKING:
    from ccproxy.observability.context import RequestContext
    from ccproxy.streaming.interfaces import IStreamingMetricsCollector

from .format_adapter import CodexFormatAdapter
from .transformers import CodexRequestTransformer, CodexResponseTransformer


logger = get_plugin_logger()


class CodexAdapter(BaseAdapter):
    """Codex adapter using ProxyService delegation pattern.

    This adapter follows the same pattern as Claude API adapter,
    delegating actual HTTP operations to ProxyService.
    """

    def __init__(
        self,
        proxy_service: IRequestHandler | None,
        auth_manager: AuthManager,
        detection_service: Any,
        http_client: Any | None = None,
        logger: Any = None,
        context: Any | None = None,  # Can be dict or PluginContext TypedDict
    ):
        """Initialize the Codex adapter.

        Args:
            proxy_service: Request handler for processing requests (can be None, will be set later)
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for Codex CLI detection
            http_client: Not used directly (for interface compatibility)
            logger: Structured logger instance
            context: Optional plugin context containing plugin_registry and other services
        """
        self.logger = logger or get_plugin_logger()
        self.proxy_service = proxy_service
        self._auth_manager = auth_manager
        self._detection_service = detection_service
        self.context = context or {}

        # Initialize components
        self.format_adapter = CodexFormatAdapter()

        # Initialize HTTP handler and transformers
        self._http_handler: PluginHTTPHandler | None = None
        self.request_transformer: CodexRequestTransformer | None = None
        self.response_transformer: CodexResponseTransformer | None = None

        # Complete initialization if proxy_service is available
        if proxy_service:
            self._complete_initialization()

    def _complete_initialization(self) -> None:
        """Complete initialization with proxy_service dependencies."""
        if not self.proxy_service:
            return

        # Type check for ProxyService specific attributes
        from ccproxy.services.proxy_service import ProxyService

        if isinstance(self.proxy_service, ProxyService):
            # Initialize HTTP handler with shared HTTP client from proxy service
            shared_client = getattr(self.proxy_service, "http_client", None)
            if not shared_client:
                raise RuntimeError("ProxyService must have http_client attribute")
            request_tracer = getattr(self.proxy_service, "request_tracer", None)
            self._http_handler = PluginHTTPHandler(
                http_client=shared_client, request_tracer=request_tracer
            )

            # Initialize transformers
            self.request_transformer = CodexRequestTransformer(self._detection_service)

            # Initialize response transformer with CORS settings
            cors_settings = (
                getattr(self.proxy_service.config, "cors", None)
                if self.proxy_service
                else None
            )
            self.response_transformer = CodexResponseTransformer(cors_settings)
        else:
            # No ProxyService available
            raise RuntimeError("CodexAdapter requires a ProxyService instance")

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
            self.logger.debug("failed_to_get_pricing_service", error=str(e))
            return None

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

        # Extract session_id
        session_id = kwargs.get("session_id") or str(uuid.uuid4())

        # Read request body
        body = await request.body()

        # Check if format conversion is needed based on endpoint
        # OpenAI format endpoints need conversion to Codex format
        needs_conversion = endpoint.endswith(
            OPENAI_CHAT_COMPLETIONS_PATH
        ) or endpoint.endswith(OPENAI_COMPLETIONS_PATH)

        # Get authentication token
        if not self._auth_manager:
            raise HTTPException(
                status_code=503, detail="Authentication manager not available"
            )

        # Get access token directly from auth manager
        access_token = await self._auth_manager.get_access_token()

        # Build auth headers with Bearer token
        auth_headers = {"Authorization": f"Bearer {access_token}"}

        # Build target URL
        target_url = f"{CODEX_API_BASE_URL}{CODEX_RESPONSES_ENDPOINT}"

        # Get RequestContext - it must exist when called via ProxyService
        from ccproxy.observability.context import RequestContext

        request_context: RequestContext | None = RequestContext.get_current()
        if not request_context:
            raise HTTPException(
                status_code=500,
                detail="RequestContext not available - plugin must be called via ProxyService",
            )

        # Create metrics collector for this request with cost calculation capability
        metrics_collector: IStreamingMetricsCollector | None = None
        if request_context:
            from .streaming_metrics import CodexStreamingMetricsCollector

            request_id = getattr(request_context, "request_id", None)
            # Get pricing service for cost calculation
            pricing_service = self._get_pricing_service()

            # Create enhanced metrics collector with pricing capability
            # The collector will extract the model from the streaming chunks
            metrics_collector = CodexStreamingMetricsCollector(
                request_id=request_id, pricing_service=pricing_service
            )

        # Create simplified provider context
        context = HandlerConfig(
            request_adapter=self.format_adapter if needs_conversion else None,
            response_adapter=self.format_adapter if needs_conversion else None,
            request_transformer=self.request_transformer,
            response_transformer=self.response_transformer,
            supports_streaming=True,
            metrics_collector=metrics_collector,
        )

        # Prepare request using HTTP handler
        if not self._http_handler:
            raise HTTPException(status_code=503, detail="HTTP handler not initialized")

        (
            transformed_body,
            headers,
            is_streaming,
        ) = await self._http_handler.prepare_request(
            request_body=body,
            handler_config=context,
            auth_headers=auth_headers,
            request_headers=dict(request.headers),
            session_id=session_id,
            access_token=access_token,
        )

        # Parse request body for model extraction
        parsed_body = {}
        if body:
            try:
                parsed_body = json.loads(body)
            except json.JSONDecodeError:
                parsed_body = {}

        self.logger.info(
            "plugin_request",
            plugin="codex",
            endpoint=endpoint,
            model=parsed_body.get("model") if isinstance(parsed_body, dict) else None,
            is_streaming=is_streaming,
            needs_conversion=needs_conversion,
            session_id=session_id,
            target_url=target_url,
        )

        # Update context with codex specific metadata
        request_context.metadata.update(
            {
                "provider": "codex",
                "service_type": "codex",
                "endpoint": endpoint.rstrip("/").split("/")[-1]
                if endpoint
                else "responses",
                "model": parsed_body.get("model", "unknown"),
                "stream": is_streaming,
                "needs_conversion": needs_conversion,
            }
        )

        # Make the actual HTTP request using the shared handler
        if not self.proxy_service:
            raise HTTPException(status_code=503, detail="Proxy service not available")

        # Get streaming handler if available
        from ccproxy.services.proxy_service import ProxyService

        streaming_handler = None
        if is_streaming and isinstance(self.proxy_service, ProxyService):
            streaming_handler = self.proxy_service.streaming_handler

        response = await self._http_handler.handle_request(
            method=method,
            url=target_url,
            headers=headers,
            body=transformed_body,
            handler_config=context,
            is_streaming=is_streaming,
            streaming_handler=streaming_handler,
            request_context=request_context,
        )

        # For deferred streaming responses, return directly (metrics collector already has cost calculation)
        from ccproxy.streaming.deferred_streaming import DeferredStreaming

        if isinstance(response, DeferredStreaming):
            self.logger.debug(
                "codex_using_deferred_response",
                response_type=type(response).__name__,
                category="http",
            )
            return response

        # For regular streaming responses, wrap to accumulate chunks and extract headers
        if is_streaming and isinstance(response, StreamingResponse):
            return await self._wrap_streaming_response(response, request_context)

        # For non-streaming responses, extract usage data if available
        if not is_streaming and hasattr(response, "body"):
            # Get response body (might be bytes or memoryview)
            response_body = response.body
            if isinstance(response_body, memoryview):
                response_body = bytes(response_body)
            await self._extract_usage_from_response(response_body, request_context)

        return response

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

    async def _extract_usage_from_response(
        self, body: bytes | str, request_context: "RequestContext"
    ) -> None:
        """Extract usage data from response body and update context.

        Common function used by both streaming and non-streaming responses.

        Args:
            body: Response body (bytes or string)
            request_context: Request context to update with usage data
        """
        try:
            import json

            # Convert body to string if needed
            body_str = body
            if isinstance(body_str, bytes):
                body_str = body_str.decode("utf-8")

            # Parse response to extract usage
            response_data = json.loads(body_str)
            usage = response_data.get("usage", {})

            if not usage:
                return

            # Extract OpenAI-specific usage fields
            # OpenAI uses prompt_tokens and completion_tokens
            tokens_input = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
            tokens_output = usage.get("completion_tokens", 0) or usage.get(
                "output_tokens", 0
            )

            # Check for cached tokens in input_tokens_details
            cache_read_tokens = 0
            if "input_tokens_details" in usage:
                cache_read_tokens = usage["input_tokens_details"].get(
                    "cached_tokens", 0
                )

            # Check for reasoning tokens in output_tokens_details
            reasoning_tokens = 0
            if "output_tokens_details" in usage:
                reasoning_tokens = usage["output_tokens_details"].get(
                    "reasoning_tokens", 0
                )

            # Calculate cost using pricing service if available
            cost_usd = None
            pricing_service = self._get_pricing_service()
            self.logger.debug(
                "pricing_service_check",
                has_pricing_service=pricing_service is not None,
                source="non_streaming",
            )
            if pricing_service:
                try:
                    model = request_context.metadata.get(
                        "model", response_data.get("model", "gpt-3.5-turbo")
                    )
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
                        cache_write_tokens=0,  # OpenAI doesn't have cache write tokens
                    )
                    cost_usd = float(cost_decimal)
                    self.logger.debug(
                        "cost_calculated",
                        model=model,
                        cost_usd=cost_usd,
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                    )
                except ModelPricingNotFoundError as e:
                    self.logger.warning(
                        "model_pricing_not_found",
                        model=model,
                        message=str(e),
                        tokens_input=tokens_input,
                        tokens_output=tokens_output,
                    )
                except PricingDataNotLoadedError as e:
                    self.logger.warning(
                        "pricing_data_not_loaded",
                        model=model,
                        message=str(e),
                    )
                except PricingServiceDisabledError as e:
                    self.logger.debug(
                        "pricing_service_disabled",
                        message=str(e),
                    )
                except Exception as e:
                    self.logger.debug(
                        "cost_calculation_failed", error=str(e), model=model
                    )

            # Update request context with usage data
            request_context.metadata.update(
                {
                    "tokens_input": tokens_input,
                    "tokens_output": tokens_output,
                    "tokens_total": tokens_input + tokens_output,
                    "cache_read_tokens": cache_read_tokens,
                    "cache_write_tokens": 0,  # OpenAI doesn't have cache write tokens
                    "reasoning_tokens": reasoning_tokens,
                    "cost_usd": cost_usd or 0.0,
                }
            )

            self.logger.debug(
                "usage_extracted",
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cache_read_tokens=cache_read_tokens,
                reasoning_tokens=reasoning_tokens,
                cost_usd=cost_usd,
                source="response_body",
            )

        except Exception as e:
            self.logger.debug("usage_extraction_failed", error=str(e))

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

        # Create metrics collector for usage extraction
        from .streaming_metrics import CodexStreamingMetricsCollector

        pricing_service = self._get_pricing_service()
        collector = CodexStreamingMetricsCollector(
            request_id=request_context.request_id, pricing_service=pricing_service
        )

        async def wrapped_iterator() -> AsyncIterator[bytes]:
            """Wrap the stream iterator to accumulate chunks."""
            nonlocal headers_extracted

            async for chunk in original_iterator:
                # Extract headers on first chunk (after streaming has started)
                if not headers_extracted:
                    headers_extracted = True
                    if "response_headers" in request_context.metadata:
                        response_headers = request_context.metadata["response_headers"]

                        # Extract relevant headers for logging
                        headers_for_log = {}
                        for k, v in response_headers.items():
                            k_lower = k.lower()
                            # Include OpenAI headers and request IDs
                            if k_lower.startswith("openai-"):
                                # Put OpenAI headers directly in metadata for access_logger
                                request_context.metadata[k_lower] = v
                                headers_for_log[k] = v
                            elif "request" in k_lower and "id" in k_lower:
                                headers_for_log[k] = v

                        # Also store the headers dictionary for display
                        request_context.metadata["headers"] = headers_for_log

                        self.logger.debug(
                            "codex_headers_extracted",
                            headers_count=len(headers_for_log),
                            headers=headers_for_log,
                            category="http",
                        )

                if isinstance(chunk, str | memoryview):
                    chunk = chunk.encode() if isinstance(chunk, str) else bytes(chunk)
                chunks.append(chunk)

                # Process this chunk for usage data
                chunk_str = chunk.decode("utf-8", errors="ignore")

                # Debug: Log first few chunks to see what we're processing
                if len(chunks) <= 3:
                    self.logger.debug(
                        "streaming_chunk_debug",
                        chunk_length=len(chunk_str),
                        chunk_preview=chunk_str[:200],
                        chunk_number=len(chunks),
                        request_id=request_context.request_id,
                        category="debug",
                    )

                is_final = collector.process_chunk(chunk_str)

                # Debug: Log collector state
                self.logger.debug(
                    "streaming_collector_state",
                    is_final=is_final,
                    metrics=collector.get_metrics(),
                    request_id=request_context.request_id,
                    category="debug",
                )

                # If we got final metrics, update context
                if is_final:
                    usage_metrics = collector.get_metrics()
                    if usage_metrics:
                        # Cost is already calculated in the collector
                        cost_usd = usage_metrics.get("cost_usd")

                        # Get reasoning tokens separately
                        reasoning_tokens = collector.get_reasoning_tokens() or 0

                        # Update request context with usage data using common format
                        request_context.metadata.update(
                            {
                                "tokens_input": usage_metrics.get("tokens_input", 0),
                                "tokens_output": usage_metrics.get("tokens_output", 0),
                                "tokens_total": (
                                    (usage_metrics.get("tokens_input") or 0)
                                    + (usage_metrics.get("tokens_output") or 0)
                                ),
                                "cost_usd": cost_usd or 0.0,
                                "cache_read_tokens": usage_metrics.get(
                                    "cache_read_tokens", 0
                                ),
                                "cache_write_tokens": 0,  # OpenAI doesn't have cache write
                                "reasoning_tokens": reasoning_tokens,
                            }
                        )

                        self.logger.debug(
                            "usage_extracted",
                            tokens_input=usage_metrics.get("tokens_input"),
                            tokens_output=usage_metrics.get("tokens_output"),
                            cache_read_tokens=usage_metrics.get("cache_read_tokens"),
                            reasoning_tokens=reasoning_tokens,
                            cost_usd=cost_usd,
                            source="streaming",
                        )

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

    async def cleanup(self) -> None:
        """Cleanup resources when shutting down."""
        try:
            # Cleanup HTTP handler if it exists
            if self._http_handler:
                if hasattr(self._http_handler, "cleanup"):
                    await self._http_handler.cleanup()
                self._http_handler = None

            # Clear references to prevent memory leaks
            self.proxy_service = None
            self.request_transformer = None
            self.response_transformer = None

            self.logger.debug("adapter_cleanup_completed")

        except Exception as e:
            self.logger.error(
                "codex_adapter_cleanup_failed",
                error=str(e),
                exc_info=e,
            )
