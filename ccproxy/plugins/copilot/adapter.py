import uuid
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.streaming import DeferredStreaming
from ccproxy.utils.headers import (
    extract_response_headers,
    filter_request_headers,
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
        # Both /v1/messages and /v1/chat/completions map to GitHub Copilot's /chat/completions endpoint
        # Format conversion (Anthropic <-> OpenAI) is handled by the format adapter chain
        return f"{self.base_url}/chat/completions"

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
                response_adapter = self.format_registry.get_adapter(
                    "openai", "anthropic"
                )

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

    def _needs_format_conversion(self, endpoint: str) -> bool:
        """Check if this endpoint needs format conversion (Anthropic format requested)."""
        return endpoint.endswith("/v1/messages")
