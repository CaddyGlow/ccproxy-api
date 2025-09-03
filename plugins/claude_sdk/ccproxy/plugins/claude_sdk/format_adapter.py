"""Format adapter for Claude SDK plugin.

This module handles format conversion between OpenAI and Anthropic formats
for the Claude SDK plugin.
"""

from typing import Any

from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class ClaudeSDKFormatAdapter:
    """Adapter for converting between OpenAI and Anthropic message formats.

    This adapter handles the conversion of requests and responses between
    OpenAI's chat completion format and Anthropic's messages format for
    the Claude SDK plugin.
    """

    def __init__(self) -> None:
        """Initialize the format adapter."""
        self.logger = logger
        self.openai_adapter = OpenAIAdapter()

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert request from OpenAI format to Anthropic format if needed.

        Args:
            request_data: Request data that may be in OpenAI format

        Returns:
            Request data in Anthropic format
        """
        # Check if this is OpenAI format (has 'messages' with role/content structure)
        if "messages" in request_data:
            # Check if it's already in Anthropic format or needs conversion
            messages = request_data.get("messages", [])
            if messages and isinstance(messages[0], dict):
                first_msg = messages[0]
                # OpenAI format has 'role' and 'content' at top level
                # Anthropic format has 'role' and 'content' where content is list of blocks
                if "role" in first_msg and isinstance(first_msg.get("content"), str):
                    # This looks like OpenAI format, convert it
                    self.logger.debug("converting_openai_format_to_anthropic_format")
                    return await self.openai_adapter.adapt_request(request_data)

        # Already in Anthropic format or not a messages request
        return request_data

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert response from Anthropic format to OpenAI format if needed.

        Args:
            response_data: Response data in Anthropic format

        Returns:
            Response data in appropriate format
        """
        # Check if we need to convert to OpenAI format
        # This is determined by the original request format (stored in context)
        # For now, we'll detect based on response structure

        if "type" in response_data and response_data["type"] == "message":
            # This is Anthropic format, check if we need OpenAI format
            # The decision should be based on the original request format
            # For now, we'll return as-is and let the caller decide
            self.logger.debug("response_in_anthropic_format")

        return response_data

    async def adapt_streaming_response(
        self, chunk: dict[str, Any], needs_openai_format: bool = False
    ) -> dict[str, Any]:
        """Convert streaming response chunk between formats if needed.

        Args:
            chunk: Streaming chunk in Anthropic format
            needs_openai_format: Whether to convert to OpenAI format

        Returns:
            Chunk in appropriate format
        """
        if needs_openai_format:
            # Convert Anthropic SSE to OpenAI SSE format
            # The OpenAIAdapter doesn't have adapt_streaming_chunk, use adapt_response
            # For streaming, we need to handle this differently
            return chunk  # Pass through for now, streaming conversion is complex

        return chunk
