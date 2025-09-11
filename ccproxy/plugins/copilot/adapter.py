import uuid
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import (
    extract_request_headers,
    extract_response_headers,
    filter_request_headers,
    filter_response_headers,
)

from .config import CopilotConfig
from .oauth.provider import CopilotOAuthProvider


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """Simplified Copilot adapter."""

    def __init__(
        self, oauth_provider: CopilotOAuthProvider, config: CopilotConfig, **kwargs: Any
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.oauth_provider = oauth_provider

        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth token
        access_token = await self.oauth_provider.ensure_copilot_token()

        # Filter headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)

        # Add Copilot headers (lowercase keys)
        copilot_headers = {}
        for key, value in self.config.api_headers.items():
            copilot_headers[key.lower()] = value

        copilot_headers["authorization"] = f"Bearer {access_token}"
        copilot_headers["x-request-id"] = str(uuid.uuid4())

        # Merge headers
        final_headers = {}
        final_headers.update(filtered_headers)
        final_headers.update(copilot_headers)

        logger.debug("copilot_request_prepared", header_count=len(final_headers))

        return body, final_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response | StreamingResponse | DeferredStreaming:
        """Process provider response with format conversion support."""
        response_headers = extract_response_headers(response)

        # Check if this is a streaming response
        content_type = response_headers.get("content-type", "")
        is_streaming = (
            "text/event-stream" in content_type or "stream" in content_type.lower()
        )

        if is_streaming:
            logger.debug(
                "copilot_streaming_response_detected",
                content_type=content_type,
                endpoint=endpoint,
                category="streaming_conversion",
            )

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
        )

    async def _create_streaming_response(
        self, response: httpx.Response, endpoint: str
    ) -> DeferredStreaming:
        """Create streaming response with format conversion support."""

        # Check if format conversion is needed based on endpoint
        needs_conversion = self._needs_format_conversion(endpoint)
        response_adapter = None

        if needs_conversion and self.format_registry:
            try:
                # Get the response adapter (openai -> anthropic) for streaming conversion
                response_adapter = self.format_registry.get("openai", "anthropic")

                logger.debug(
                    "copilot_format_adapter_loaded",
                    endpoint=endpoint,
                    has_response_adapter=bool(response_adapter),
                    category="streaming_conversion",
                )
            except Exception as e:
                logger.warning(
                    "copilot_format_adapter_loading_failed",
                    error=str(e),
                    endpoint=endpoint,
                    category="streaming_conversion",
                )

        # Get HTTP client from pool manager
        client = await self.http_pool_manager.get_client()

        # Create minimal HandlerConfig only for format conversion
        handler_config = None
        if response_adapter:
            from ccproxy.services.handler_config import HandlerConfig

            handler_config = HandlerConfig(
                response_adapter=response_adapter,
                supports_streaming=True,
            )
            logger.debug(
                "copilot_minimal_handler_config_created",
                endpoint=endpoint,
                from_format="openai",
                to_format="anthropic",
                category="streaming_conversion",
            )

        # Create DeferredStreaming with minimal handler config for format conversion
        return DeferredStreaming(
            method="POST",
            url=str(response.url),
            headers=dict(response.request.headers),
            body=response.request.content,
            client=client,
            handler_config=handler_config,
        )

    async def handle_request_gh_api(self, request: Request) -> Response:
        """Forward request to GitHub API with proper authentication.

        Args:
            path: API path (e.g., '/copilot_internal/user')
            mode: API mode - 'api' for GitHub API with OAuth token, 'copilot' for Copilot API with Copilot token
            method: HTTP method
            body: Request body
            extra_headers: Additional headers
        """
        access_token = await self.oauth_provider.ensure_oauth_token()
        base_url = "https://api.github.com"

        headers = {
            "authorization": f"Bearer {access_token}",
            "accept": "application/json",
        }
        # Get context from middleware (already initialized)
        ctx = request.state.context

        # Step 1: Extract request data
        body = await request.body()
        headers = extract_request_headers(request)
        method = request.method
        endpoint = ctx.metadata.get("endpoint", "")
        target_url = f"{base_url}{endpoint}"

        provider_response = await self._execute_http_request(
            method,
            target_url,
            headers,
            body,
        )

        filtered_headers = filter_response_headers(dict(provider_response.headers))

        return Response(
            content=provider_response.content,
            status_code=provider_response.status_code,
            headers=filtered_headers,
            media_type=provider_response.headers.get(
                "content-type", "application/json"
            ),
        )

    def _needs_format_conversion(self, endpoint: str) -> bool:
        """Decide if format conversion is needed for a given endpoint."""
        return False
