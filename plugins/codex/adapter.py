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
    ):
        """Initialize the Codex adapter.

        Args:
            proxy_service: Request handler for processing requests (can be None, will be set later)
            auth_manager: Authentication manager for credentials
            detection_service: Detection service for Codex CLI detection
            http_client: Not used directly (for interface compatibility)
            logger: Structured logger instance
        """
        self.logger = logger or get_plugin_logger()
        self.proxy_service = proxy_service
        self._auth_manager = auth_manager
        self._detection_service = detection_service

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

        # Create metrics collector for this request
        metrics_collector: IStreamingMetricsCollector | None = None
        if request_context:
            from .streaming_metrics import CodexStreamingMetricsCollector

            request_id = getattr(request_context, "request_id", None)
            metrics_collector = CodexStreamingMetricsCollector(request_id=request_id)

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
            target_url=target_url,
            needs_conversion=needs_conversion,
            is_streaming=is_streaming,
            session_id=session_id,
            model=parsed_body.get("model") if isinstance(parsed_body, dict) else None,
            category="http",
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

        # For deferred streaming responses, return as-is
        from ccproxy.streaming.deferred_streaming import DeferredStreaming

        if isinstance(response, DeferredStreaming):
            self.logger.debug(
                "codex_using_deferred_response",
                response_type=type(response).__name__,
                category="http",
            )
            return response

        # For other responses, return as-is
        # DeferredStreaming already handles metrics collection
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
