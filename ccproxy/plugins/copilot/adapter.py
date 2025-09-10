import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.responses import Response, StreamingResponse

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import (
    extract_response_headers,
    filter_request_headers,
    to_canonical_headers,
)

from .config import CopilotConfig
from .oauth.provider import CopilotOAuthProvider


logger = get_plugin_logger()


class CopilotAdapter(BaseHTTPAdapter):
    """Simplified Copilot adapter."""

    def __init__(
        self, oauth_provider: CopilotOAuthProvider, config: CopilotConfig, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.oauth_provider = oauth_provider
        self.config = config

    def _get_base_url(self) -> str:
        """Get base URL from Copilot config."""
        return self.config.get_base_url()

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
    ) -> Response | StreamingResponse:
        response_headers = extract_response_headers(response)

        # Filter response headers
        safe_headers = {}
        for key, value in response_headers.items():
            if key not in {"connection", "transfer-encoding", "content-encoding"}:
                safe_headers[key] = value

        # Check if this is a streaming response
        content_type = response_headers.get("content-type", "")
        if "text/event-stream" in content_type or "stream" in content_type.lower():
            logger.debug(
                "copilot_streaming_response_detected", content_type=content_type
            )

            # Create streaming response
            async def stream_generator() -> AsyncIterator[bytes]:
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                content=stream_generator(),
                status_code=response.status_code,
                headers=to_canonical_headers(safe_headers),
                media_type=content_type or "text/event-stream",
            )
        else:
            # Non-streaming response
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=to_canonical_headers(safe_headers),
            )
