import json
import uuid
from typing import Any

import httpx
from starlette.responses import Response

from ccproxy.core.logging import get_plugin_logger
from ccproxy.services.adapters.http_adapter import BaseHTTPAdapter
from ccproxy.utils.headers import (
    extract_response_headers,
    filter_request_headers,
    to_canonical_headers,
)

from .detection_service import CodexDetectionService
from .models import CodexAuthData


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""

    def __init__(self, detection_service: CodexDetectionService, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.detection_service = detection_service

    async def get_target_url(self, endpoint: str) -> str:
        return "https://chat.openai.com/backend-anon/responses"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth
        auth_data = await self.auth_manager.load_credentials()

        # Parse and convert body
        body_data = json.loads(body.decode()) if body else {}

        # Format conversion
        if self._needs_format_conversion(endpoint):
            body_data = await self._convert_to_codex_format(body_data)

        # Inject instructions (after format conversion to ensure they're preserved)
        if "instructions" not in body_data or body_data.get("instructions") is None:
            body_data["instructions"] = self._get_instructions()

        if "stream" not in body_data:
            body_data["stream"] = True

        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        filtered_headers.update(
            {
                "authorization": f"Bearer {auth_data.access_token}",
                "chatgpt-account-id": auth_data.account_id,
                "session-id": str(uuid.uuid4()),
                "content-type": "application/json",
            }
        )

        # Add CLI headers
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.headers:
                cli_headers = cached_data.headers.to_headers_dict()
                for key, value in cli_headers.items():
                    filtered_headers[key.lower()] = value

        return json.dumps(body_data).encode(), filtered_headers

    async def process_provider_response(
        self, response: httpx.Response, endpoint: str
    ) -> Response:
        # Convert response format
        response_data = json.loads(response.content)
        converted_data = await self._convert_codex_to_openai(response_data)

        response_headers = extract_response_headers(response)

        return Response(
            content=json.dumps(converted_data).encode(),
            status_code=response.status_code,
            headers=to_canonical_headers(response_headers),
        )

    # Helper methods (move from transformers)
    def _needs_format_conversion(self, endpoint: str) -> bool:
        return True  # Codex always needs conversion

    def _get_instructions(self) -> str:
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field
        return "You are a coding agent..."

    async def _convert_to_codex_format(
        self, body_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI Chat Completions format to Codex Response API format.

        Args:
            body_data: OpenAI Chat Completions format request data

        Returns:
            Codex Response API format request data
        """
        from ccproxy.adapters.openai.response_adapter import ResponseAdapter

        # Preserve original instructions if they exist
        original_instructions = body_data.get("instructions")

        try:
            adapter = ResponseAdapter()
            response_request = adapter.chat_to_response_request(body_data)
            converted_data = response_request.model_dump()

            # Ensure stream=True for Codex (always required)
            converted_data["stream"] = True

            # Restore original instructions if they existed
            if original_instructions is not None:
                converted_data["instructions"] = original_instructions

            return converted_data
        except Exception as e:
            logger.error(
                "chat_to_codex_conversion_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback - return original data with stream=True
            body_data["stream"] = True
            return body_data

    async def _convert_codex_to_openai(
        self, response_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Codex Response API format to OpenAI Chat Completions format.

        Args:
            response_data: Codex Response API format response data

        Returns:
            OpenAI Chat Completions format response data
        """
        from ccproxy.adapters.openai.response_adapter import ResponseAdapter

        try:
            adapter = ResponseAdapter()
            chat_completion = adapter.response_to_chat_completion(response_data)
            return chat_completion.model_dump()
        except Exception as e:
            logger.error(
                "codex_to_chat_conversion_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback - return original data
            return response_data
