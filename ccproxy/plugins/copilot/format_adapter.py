"""Format adapter for OpenAI ↔ Copilot format conversion."""

from collections.abc import AsyncIterator
from typing import Any

from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionRequest,
    OpenAIMessage,
)
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class OpenAIToCopilotAdapter:
    """Convert OpenAI format requests/responses to Copilot format."""

    async def adapt_request(self, openai_request: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI chat completion request to Copilot format.

        Args:
            openai_request: OpenAI format request

        Returns:
            Copilot format request
        """
        logger.debug(
            "adapting_openai_request",
            model=openai_request.get("model"),
            stream=openai_request.get("stream", False),
        )

        # Convert to OpenAI request object for validation
        openai_request_obj = OpenAIChatCompletionRequest.model_validate(openai_request)

        # Copilot uses the same format as OpenAI, so just return the validated data
        result = openai_request_obj.model_dump(exclude_none=True)

        logger.debug(
            "openai_request_adapted",
            messages_count=len(result.get("messages", [])),
            model=result.get("model"),
        )

        return result

    async def adapt_response(self, copilot_response: dict[str, Any]) -> dict[str, Any]:
        """Convert Copilot response to OpenAI format.

        Args:
            copilot_response: Copilot format response

        Returns:
            OpenAI format response
        """
        logger.debug(
            "adapting_copilot_response",
            object_type=copilot_response.get("object"),
            choices_count=len(copilot_response.get("choices", [])),
        )

        # Copilot responses are already in OpenAI format for the most part
        # Just ensure proper structure and add any missing fields
        result = {
            "id": copilot_response.get("id"),
            "object": copilot_response.get("object", "chat.completion"),
            "created": copilot_response.get("created"),
            "model": copilot_response.get("model"),
            "choices": copilot_response.get("choices", []),
            "usage": copilot_response.get("usage"),
        }

        # Clean up None values
        result = {k: v for k, v in result.items() if v is not None}

        logger.debug(
            "copilot_response_adapted",
            object_type=result.get("object"),
        )

        return result

    async def adapt_stream_chunk(self, copilot_chunk: dict[str, Any]) -> dict[str, Any]:
        """Convert Copilot streaming chunk to OpenAI format.

        Args:
            copilot_chunk: Copilot format streaming chunk

        Returns:
            OpenAI format streaming chunk
        """
        # Copilot chunks are already in OpenAI streaming format
        # Ensure proper structure
        result = {
            "id": copilot_chunk.get("id"),
            "object": copilot_chunk.get("object", "chat.completion.chunk"),
            "created": copilot_chunk.get("created"),
            "model": copilot_chunk.get("model"),
            "choices": copilot_chunk.get("choices", []),
        }

        # Clean up None values
        result = {k: v for k, v in result.items() if v is not None}

        return result


class CopilotToOpenAIAdapter:
    """Convert Copilot format requests/responses to OpenAI format."""

    async def adapt_request(self, copilot_request: dict[str, Any]) -> dict[str, Any]:
        """Convert Copilot request to OpenAI format.

        Args:
            copilot_request: Copilot format request

        Returns:
            OpenAI format request
        """
        logger.debug(
            "adapting_copilot_request",
            model=copilot_request.get("model"),
            stream=copilot_request.get("stream", False),
        )

        # Copilot requests are already in OpenAI format
        # Just validate and clean up
        openai_request = {
            "messages": copilot_request.get("messages", []),
            "model": copilot_request.get("model", "gpt-4"),
            "stream": copilot_request.get("stream", False),
        }

        # Add optional parameters if present
        optional_params = [
            "temperature",
            "max_tokens",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "top_p",
            "n",
            "user",
        ]

        for param in optional_params:
            if param in copilot_request:
                openai_request[param] = copilot_request[param]

        logger.debug(
            "copilot_request_adapted",
            messages_count=len(openai_request.get("messages", [])),
        )

        return openai_request

    async def adapt_response(self, openai_response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI response to Copilot format.

        Args:
            openai_response: OpenAI format response

        Returns:
            Copilot format response
        """
        logger.debug(
            "adapting_openai_response",
            object_type=openai_response.get("object"),
        )

        # OpenAI and Copilot response formats are compatible
        # Preserve all original fields and add any missing standard fields
        result = dict(openai_response)  # Start with all original fields

        # Ensure standard fields have default values if missing
        result.setdefault("object", "chat.completion")

        # For streaming chunks, extract finish_reason from choices if present
        if result.get("object") == "chat.completion.chunk":
            choices = result.get("choices", [])
            if choices and len(choices) > 0:
                finish_reason = choices[0].get("finish_reason")
                if finish_reason:
                    result["finish_reason"] = finish_reason

        return result

    async def adapt_stream_chunk(self, openai_chunk: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI streaming chunk to Copilot format.

        Args:
            openai_chunk: OpenAI format streaming chunk

        Returns:
            Copilot format streaming chunk
        """
        logger.debug(
            "adapting_openai_stream_chunk",
            object_type=openai_chunk.get("object"),
        )

        # OpenAI and Copilot streaming formats are compatible
        # Preserve all original fields and add any missing standard fields
        result = dict(openai_chunk)  # Start with all original fields

        # Ensure standard fields have default values if missing
        result.setdefault("object", "chat.completion.chunk")

        # Extract finish_reason from choices if present
        choices = result.get("choices", [])
        if choices and len(choices) > 0:
            finish_reason = choices[0].get("finish_reason")
            if finish_reason:
                result["finish_reason"] = finish_reason

        return result


class CopilotFormatAdapter:
    """Bidirectional format adapter for OpenAI ↔ Copilot conversion."""

    def __init__(self) -> None:
        """Initialize the format adapter."""
        self.openai_to_copilot = OpenAIToCopilotAdapter()
        self.copilot_to_openai = CopilotToOpenAIAdapter()

    async def adapt_request(
        self, request: dict[str, Any], from_format: str, to_format: str
    ) -> dict[str, Any]:
        """Adapt request between formats.

        Args:
            request: Request data
            from_format: Source format (openai or copilot)
            to_format: Target format (openai or copilot)

        Returns:
            Adapted request
        """
        if from_format == to_format:
            return request

        if from_format == "openai" and to_format == "copilot":
            return await self.openai_to_copilot.adapt_request(request)
        elif from_format == "copilot" and to_format == "openai":
            return await self.copilot_to_openai.adapt_request(request)
        else:
            logger.warning(
                "unsupported_format_conversion",
                from_format=from_format,
                to_format=to_format,
            )
            return request

    async def adapt_response(
        self, response: dict[str, Any], from_format: str, to_format: str
    ) -> dict[str, Any]:
        """Adapt response between formats.

        Args:
            response: Response data
            from_format: Source format (openai or copilot)
            to_format: Target format (openai or copilot)

        Returns:
            Adapted response
        """
        if from_format == to_format:
            return response

        if (
            from_format == "copilot"
            and to_format == "openai"
            or from_format == "openai"
            and to_format == "copilot"
        ):
            return await self.copilot_to_openai.adapt_response(response)
        else:
            logger.warning(
                "unsupported_format_conversion",
                from_format=from_format,
                to_format=to_format,
            )
            return response

    async def adapt_stream_chunk(
        self, chunk: dict[str, Any], from_format: str, to_format: str
    ) -> dict[str, Any]:
        """Adapt streaming chunk between formats.

        Args:
            chunk: Streaming chunk data
            from_format: Source format (openai or copilot)
            to_format: Target format (openai or copilot)

        Returns:
            Adapted chunk
        """
        if from_format == to_format:
            return chunk

        if from_format == "copilot" and to_format == "openai":
            return await self.copilot_to_openai.adapt_stream_chunk(chunk)
        elif from_format == "openai" and to_format == "copilot":
            # For now, OpenAI to Copilot streaming is the same format
            return chunk
        else:
            logger.warning(
                "unsupported_stream_format_conversion",
                from_format=from_format,
                to_format=to_format,
            )
            return chunk

    async def process_stream(
        self, stream: AsyncIterator[dict[str, Any]], from_format: str, to_format: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Process streaming response with format conversion.

        Args:
            stream: Input stream
            from_format: Source format
            to_format: Target format

        Yields:
            Adapted streaming chunks
        """
        async for chunk in stream:
            adapted_chunk = await self.adapt_stream_chunk(chunk, from_format, to_format)
            yield adapted_chunk
