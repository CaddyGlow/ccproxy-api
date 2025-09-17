"""Format adapter for Claude SDK plugin.

This module handles format conversion between OpenAI and Anthropic formats
for the Claude SDK plugin.
"""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.core.logging import get_plugin_logger
from ccproxy.llms.adapters.formatter_adapter import FormatterRegistryAdapter


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
        # Create FormatterRegistryAdapter with registry
        from ccproxy.llms.adapters.formatter_registry import (
            FormatterRegistry,
            iter_registered_formatters,
            load_builtin_formatter_modules,
        )

        registry = FormatterRegistry()
        load_builtin_formatter_modules()
        for registration in iter_registered_formatters():
            registry.register(
                source_format=registration.source_format,
                target_format=registration.target_format,
                operation=registration.operation,
                formatter=registration.formatter,
            )

        self.formatter_adapter = FormatterRegistryAdapter(
            formatter_registry=registry,
            source_format="openai",
            target_format="anthropic"
        )

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
                    from ccproxy.llms.adapters.formatter_adapter import (
                        FormatterGenericModel,
                    )
                    generic_request = FormatterGenericModel(**request_data)
                    result = await self.formatter_adapter.adapt_request(generic_request)
                    return result.model_dump()

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

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert streaming response, passing through Anthropic format chunks.

        Args:
            stream: Stream of Anthropic message chunks

        Yields:
            Anthropic format dict objects (SSE formatting handled by streaming system)
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of stream adaptation - pass through dict objects."""
        async for chunk in stream:
            if isinstance(chunk, dict):
                yield chunk
