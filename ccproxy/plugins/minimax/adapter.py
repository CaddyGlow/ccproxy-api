"""HTTP adapter for the MiniMax provider plugin."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
from starlette.responses import Response

from ccproxy.core.errors import AuthenticationError
from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import extract_response_headers, filter_request_headers

from .config import MiniMaxConfig


logger = get_plugin_logger()


class MiniMaxAdapter(BaseHTTPAdapter):
    """MiniMax adapter using static API-key (Bearer) authentication."""

    def __init__(
        self,
        config: MiniMaxConfig | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config=config or MiniMaxConfig(), **kwargs)
        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        return f"{self.base_url}/{endpoint.lstrip('/')}"

    def _resolve_api_key(self) -> str:
        api_key = getattr(self.config, "api_key", None)
        if not api_key:
            logger.warning(
                "minimax_api_key_missing",
                category="auth",
            )
            raise AuthenticationError(
                "MiniMax API key is not configured. Set the 'api_key' option for "
                "the minimax plugin."
            )
        return str(api_key)

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        api_key = self._resolve_api_key()

        # Drop any inbound client credentials before adding our own.
        filtered_headers = filter_request_headers(headers, preserve_auth=False)

        provider_headers = {
            key.lower(): str(value)
            for key, value in self.config.api_headers.items()
            if value is not None
        }
        provider_headers["authorization"] = f"Bearer {api_key}"
        provider_headers["x-request-id"] = str(uuid.uuid4())

        final_headers = {**filtered_headers, **provider_headers}

        logger.debug("minimax_request_prepared", header_count=len(final_headers))

        return body, final_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response:
        """Return the upstream response verbatim.

        Streaming detection and format-chain conversion are handled centrally
        by ``BaseHTTPAdapter``; non-streaming responses are forwarded as-is.
        """
        response_headers = extract_response_headers(response)
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=response_headers,
            media_type=response.headers.get("content-type"),
        )
