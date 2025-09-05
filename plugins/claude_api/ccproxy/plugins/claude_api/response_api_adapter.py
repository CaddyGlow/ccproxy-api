"""Composite adapter for Response API to Anthropic format conversion.

This adapter chains existing adapters to provide Response API ↔ Anthropic format conversion:
1. ResponseAdapter: Response API ↔ OpenAI Chat Completions
2. OpenAIAdapter: OpenAI Chat Completions ↔ Anthropic Messages
"""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.adapters.openai.adapter import OpenAIAdapter
from ccproxy.adapters.openai.response_adapter import ResponseAdapter
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class ResponseAPIAnthropicAdapter(APIAdapter):
    """Composite adapter for Response API ↔ Anthropic Messages conversion.

    This adapter provides conversion between Response API and Anthropic Messages format
    by chaining two existing, well-tested adapters:
    - ResponseAdapter: Handles Response API ↔ OpenAI Chat Completions conversion
    - OpenAIAdapter: Handles OpenAI Chat Completions ↔ Anthropic Messages conversion
    """

    def __init__(self) -> None:
        """Initialize the composite adapter with chained adapters."""
        # ResponseAdapter handles Response API ↔ OpenAI Chat Completions
        self.response_adapter = ResponseAdapter()

        # OpenAIAdapter handles OpenAI Chat Completions ↔ Anthropic Messages
        self.openai_adapter = OpenAIAdapter()

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API request to Anthropic Messages format.

        Flow: Response API → OpenAI Chat Completions → Anthropic Messages

        Args:
            request: Response API request

        Returns:
            Anthropic Messages formatted request
        """
        try:
            # Step 1: Convert Response API → OpenAI Chat Completions REQUEST
            # We need to reverse the logic and extract request data from Response API format
            openai_request = self._response_api_to_openai_request(request)

            logger.debug(
                "response_api_adapter_step1_completed",
                conversion="response_api_to_openai",
                openai_keys=list(openai_request.keys()),
                has_messages=bool(openai_request.get("messages")),
            )

            # Step 2: Convert OpenAI Chat Completions → Anthropic Messages
            anthropic_request = await self.openai_adapter.adapt_request(openai_request)

            logger.debug(
                "response_api_adapter_step2_completed",
                conversion="openai_to_anthropic",
                anthropic_keys=list(anthropic_request.keys()),
                has_messages=bool(anthropic_request.get("messages")),
                has_max_tokens=bool(anthropic_request.get("max_tokens")),
            )

            return anthropic_request

        except Exception as e:
            logger.error(
                "response_api_adapter_request_conversion_failed",
                error=str(e),
                request_keys=list(request.keys())
                if isinstance(request, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic Messages response to Response API format.

        Flow: Anthropic Messages → OpenAI Chat Completions → Response API

        Args:
            response: Anthropic Messages response

        Returns:
            Response API formatted response
        """
        try:
            # Step 1: Convert Anthropic Messages → OpenAI Chat Completions
            openai_response = await self.openai_adapter.adapt_response(response)

            logger.debug(
                "response_api_adapter_response_step1_completed",
                conversion="anthropic_to_openai",
                openai_keys=list(openai_response.keys()),
                choices_count=len(openai_response.get("choices", [])),
            )

            # Step 2: Convert OpenAI Chat Completions → Response API
            # Use ResponseAdapter's chat_to_response_request method
            response_request = self.response_adapter.chat_to_response_request(
                openai_response
            )
            response_api_response = response_request.model_dump()

            logger.debug(
                "response_api_adapter_response_step2_completed",
                conversion="openai_to_response_api",
                response_keys=list(response_api_response.keys()),
                model=response_api_response.get("model"),
            )

            return response_api_response

        except Exception as e:
            logger.error(
                "response_api_adapter_response_conversion_failed",
                error=str(e),
                response_keys=list(response.keys())
                if isinstance(response, dict)
                else "not_dict",
                exc_info=e,
            )
            raise

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert streaming response between Response API and Anthropic formats.

        Args:
            stream: Streaming response data

        Yields:
            Converted streaming response chunks
        """
        logger.info("response_api_anthropic_adapter_adapt_stream_called")
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of streaming conversion - return dict objects."""
        try:
            logger.debug("response_api_adapter_stream_conversion_started")

            # For streaming, we need to handle the conversion differently
            # This is a simplified implementation - full streaming conversion
            # would require more complex state management
            async for chunk in stream:
                if isinstance(chunk, dict):
                    yield chunk

            logger.debug("response_api_adapter_stream_conversion_completed")

        except Exception as e:
            logger.error(
                "response_api_adapter_stream_conversion_failed",
                error=str(e),
                exc_info=e,
            )
            raise

    def _response_api_to_openai_request(
        self, response_api_request: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert Response API request to OpenAI Chat Completions request format.

        Response API format:
        {
            "model": "claude-3-5-sonnet-20241022",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}]
                }
            ],
            "max_completion_tokens": 50
        }

        OpenAI format:
        {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": "Hello!"
                }
            ],
            "max_tokens": 50
        }
        """
        openai_request = {}

        # Copy basic fields
        if "model" in response_api_request:
            openai_request["model"] = response_api_request["model"]

        # Convert input to messages and flatten content
        if "input" in response_api_request:
            messages = []
            for msg in response_api_request["input"]:
                openai_msg = {"role": msg.get("role", "user")}

                # Flatten content from Response API structured format to simple string
                content = msg.get("content", [])
                if isinstance(content, list):
                    # Extract text from content blocks
                    text_content = ""
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                    openai_msg["content"] = text_content
                elif isinstance(content, str):
                    openai_msg["content"] = content
                else:
                    openai_msg["content"] = str(content)

                messages.append(openai_msg)
            openai_request["messages"] = messages

        # Convert token limits
        if "max_completion_tokens" in response_api_request:
            openai_request["max_tokens"] = response_api_request["max_completion_tokens"]
        elif "max_tokens" in response_api_request:
            openai_request["max_tokens"] = response_api_request["max_tokens"]

        # Copy other optional fields
        for field in ["temperature", "top_p", "stream", "tools", "tool_choice"]:
            if field in response_api_request:
                openai_request[field] = response_api_request[field]

        return openai_request
