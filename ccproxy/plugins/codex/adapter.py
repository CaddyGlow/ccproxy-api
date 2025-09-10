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


logger = get_plugin_logger()


class CodexAdapter(BaseHTTPAdapter):
    """Simplified Codex adapter."""

    def __init__(self, detection_service: CodexDetectionService, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.detection_service = detection_service
        self.base_url = self.config.base_url.rstrip("/")

    async def get_target_url(self, endpoint: str) -> str:
        # Old URL: https://chat.openai.com/backend-anon/responses (308 redirect)
        return f"{self.base_url}/responses"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth credentials and profile
        auth_data = await self.auth_manager.load_credentials()
        if not auth_data:
            raise ValueError("No authentication credentials available")

        # Get profile to extract chatgpt_account_id
        profile = await self.auth_manager.get_profile_quick()
        chatgpt_account_id = profile.chatgpt_account_id if profile else None

        # Parse body (format conversion is now handled by format chain)
        body_data = json.loads(body.decode()) if body else {}

        # Inject instructions if not present
        if "instructions" not in body_data or body_data.get("instructions") is None:
            body_data["instructions"] = self._get_instructions()

        # Ensure stream is enabled (required by Codex backend)
        if "stream" not in body_data:
            body_data["stream"] = True

        # Remove any prefixed metadata fields that shouldn't be sent to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter and add headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        base_headers = {
            "authorization": f"Bearer {auth_data.access_token}",
            "session_id": str(uuid.uuid4()),
            "content-type": "application/json",
        }
        # Add chatgpt-account-id only if available
        if chatgpt_account_id is not None:
            base_headers["chatgpt-account-id"] = chatgpt_account_id

        filtered_headers.update(base_headers)

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
        # Check if response is JSON before parsing
        try:
            response_data = json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(
                "invalid_json_response",
                status_code=response.status_code,
                content_type=response.headers.get("content-type"),
                content_preview=response.content[:200] if response.content else None,
                error=str(e),
            )
            # For non-JSON responses (like redirects), return as-is
            response_headers = extract_response_headers(response)
            return Response(
                content=response.content,
                status_code=response.status_code,
                headers=response_headers,
            )

        # Response format conversion is now handled by format chain
        response_headers = extract_response_headers(response)

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=to_canonical_headers(response_headers),
        )

    # Helper methods
    def _remove_metadata_fields(self, data: dict[str, Any]) -> dict[str, Any]:
        """Remove fields that start with '_' as they are internal metadata.

        Args:
            data: Dictionary that may contain metadata fields

        Returns:
            Cleaned dictionary without metadata fields
        """
        if not isinstance(data, dict):
            return data

        # Create a new dict without keys starting with '_'
        cleaned_data: dict[str, Any] = {}
        for key, value in data.items():
            if not key.startswith("_"):
                # Recursively clean nested dictionaries
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_metadata_fields(value)
                elif isinstance(value, list):
                    # Clean list items if they are dictionaries
                    cleaned_items: list[Any] = []
                    for item in value:
                        if isinstance(item, dict):
                            cleaned_items.append(self._remove_metadata_fields(item))
                        else:
                            cleaned_items.append(item)
                    cleaned_data[key] = cleaned_items
                else:
                    cleaned_data[key] = value

        return cleaned_data

    def _get_instructions(self) -> str:
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.instructions:
                return cached_data.instructions.instructions_field
        return "You are a coding agent..."
