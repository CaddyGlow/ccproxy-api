import json
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

from .detection_service import ClaudeAPIDetectionService


logger = get_plugin_logger()


class ClaudeAPIAdapter(BaseHTTPAdapter):
    """Simplified Claude API adapter."""

    def __init__(
        self, detection_service: ClaudeAPIDetectionService, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.detection_service = detection_service

    async def get_target_url(self, endpoint: str) -> str:
        return "https://api.anthropic.com/v1/messages"

    async def prepare_provider_request(
        self, body: bytes, headers: dict[str, str], endpoint: str
    ) -> tuple[bytes, dict[str, str]]:
        # Get auth
        auth_data = await self.auth_manager.load_credentials()
        access_token = auth_data.claude_ai_oauth.access_token

        # Parse body
        body_data = json.loads(body.decode()) if body else {}

        # Inject system prompt if available
        if self.detection_service:
            cached_data = self.detection_service.get_cached_data()
            if cached_data and cached_data.system_prompt:
                body_data = self._inject_system_prompt(
                    body_data, cached_data.system_prompt
                )

        # Format conversion if needed
        if self._needs_openai_conversion(endpoint):
            body_data = await self._convert_openai_to_anthropic(body_data)

        # Remove any prefixed metadata fields that shouldn't be sent to the API
        body_data = self._remove_metadata_fields(body_data)

        # Filter headers
        filtered_headers = filter_request_headers(headers, preserve_auth=False)
        filtered_headers["authorization"] = f"Bearer {access_token.get_secret_value()}"

        # Add CLI headers if available
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
        response_headers = extract_response_headers(response)
        content = response.content

        # Format conversion if needed
        if self._needs_anthropic_conversion(endpoint):
            response_data = json.loads(content)
            converted_data = await self._convert_anthropic_to_openai(response_data)
            content = json.dumps(converted_data).encode()

        return Response(
            content=content,
            status_code=response.status_code,
            headers=to_canonical_headers(response_headers),
        )

    # Helper methods (move from transformers)
    def _inject_system_prompt(
        self, body_data: dict[str, Any], system_prompt: Any
    ) -> dict[str, Any]:
        """Inject system prompt from Claude CLI detection.

        Args:
            body_data: The request body data dict
            system_prompt: System prompt data from detection service

        Returns:
            Modified body data with system prompt injected
        """
        if not system_prompt:
            return body_data

        # Get the system field from the system prompt data
        system_field = (
            system_prompt.system_field
            if hasattr(system_prompt, "system_field")
            else system_prompt
        )

        if not system_field:
            return body_data

        # Mark the detected system prompt as injected for preservation
        marked_system = self._mark_injected_system_prompts(system_field)

        existing_system = body_data.get("system")

        if existing_system is None:
            # No existing system prompt, inject the marked detected one
            body_data["system"] = marked_system
        else:
            # Request has existing system prompt, prepend the marked detected one
            if isinstance(marked_system, list):
                if isinstance(existing_system, str):
                    # Detected is marked list, existing is string
                    body_data["system"] = marked_system + [
                        {"type": "text", "text": existing_system}
                    ]
                elif isinstance(existing_system, list):
                    # Both are lists, concatenate (detected first)
                    body_data["system"] = marked_system + existing_system
            else:
                # Convert both to list format for consistency
                if isinstance(existing_system, str):
                    body_data["system"] = [
                        {
                            "type": "text",
                            "text": str(marked_system),
                            "_ccproxy_injected": True,
                        },
                        {"type": "text", "text": existing_system},
                    ]
                elif isinstance(existing_system, list):
                    body_data["system"] = [
                        {
                            "type": "text",
                            "text": str(marked_system),
                            "_ccproxy_injected": True,
                        }
                    ] + existing_system

        return body_data

    def _mark_injected_system_prompts(self, system_data: Any) -> Any:
        """Mark system prompts as injected by ccproxy for preservation.

        Args:
            system_data: System prompt data to mark

        Returns:
            System data with injected blocks marked with _ccproxy_injected metadata
        """
        if isinstance(system_data, str):
            # String format - convert to list with marking
            return [{"type": "text", "text": system_data, "_ccproxy_injected": True}]
        elif isinstance(system_data, list):
            # List format - mark each block as injected
            marked_data = []
            for block in system_data:
                if isinstance(block, dict):
                    # Copy block and add marking
                    marked_block = block.copy()
                    marked_block["_ccproxy_injected"] = True
                    marked_data.append(marked_block)
                else:
                    # Preserve non-dict blocks as-is
                    marked_data.append(block)
            return marked_data

        return system_data

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
        cleaned_data = {}
        for key, value in data.items():
            if not key.startswith("_"):
                # Recursively clean nested dictionaries
                if isinstance(value, dict):
                    cleaned_data[key] = self._remove_metadata_fields(value)
                elif isinstance(value, list):
                    # Clean list items if they are dictionaries
                    cleaned_data[key] = [
                        self._remove_metadata_fields(item)
                        if isinstance(item, dict)
                        else item
                        for item in value
                    ]
                else:
                    cleaned_data[key] = value

        return cleaned_data

    def _needs_openai_conversion(self, endpoint: str) -> bool:
        return endpoint.endswith("/chat/completions")

    async def _convert_openai_to_anthropic(
        self, body_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI format to Anthropic format using the OpenAI adapter.

        Args:
            body_data: OpenAI format request data

        Returns:
            Anthropic format request data
        """
        from ccproxy.adapters.openai.adapter import OpenAIAdapter

        try:
            adapter = OpenAIAdapter()
            return await adapter.adapt_request(body_data)
        except Exception as e:
            logger.error(
                "openai_to_anthropic_conversion_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback - return original data
            return body_data

    def _needs_anthropic_conversion(self, endpoint: str) -> bool:
        return endpoint.endswith("/chat/completions")

    async def _convert_anthropic_to_openai(
        self, response_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Anthropic format to OpenAI format using the OpenAI adapter.

        Args:
            response_data: Anthropic format response data

        Returns:
            OpenAI format response data
        """
        from ccproxy.adapters.openai.adapter import OpenAIAdapter

        try:
            adapter = OpenAIAdapter()
            return await adapter.adapt_response(response_data)
        except Exception as e:
            logger.error(
                "anthropic_to_openai_conversion_failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            # Fallback - return original data
            return response_data
