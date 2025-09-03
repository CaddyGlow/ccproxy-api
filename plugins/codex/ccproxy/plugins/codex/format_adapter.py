"""Codex format adapter for OpenAI conversion."""

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.adapters.openai.response_adapter import ResponseAdapter
from ccproxy.core.logging import get_plugin_logger


logger = get_plugin_logger()


class CodexFormatAdapter(APIAdapter):
    """Handles format conversion between OpenAI Chat Completions and Codex Response API formats.

    This adapter delegates to the ResponseAdapter which knows how to:
    1. Convert Chat Completions → Response API format (for requests)
    2. Convert Response API → Chat Completions format (for responses)
    3. Handle SSE streaming conversion (for streaming responses)
    """

    def __init__(self) -> None:
        """Initialize the format adapter."""
        self._response_adapter = ResponseAdapter()

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Messages-based request to Response API format.

        Handles both OpenAI Chat Completions and Anthropic Messages formats.

        Args:
            request_data: OpenAI Chat Completions or Anthropic Messages request

        Returns:
            Codex Response API formatted request
        """
        if "messages" in request_data:
            # Detect format type for logging
            format_type = self._detect_request_format(request_data)
            
            # Use ResponseAdapter to convert Messages → Response API
            # This works for both OpenAI Chat Completions and Anthropic Messages
            has_tools = bool(request_data.get("tools"))
            has_tool_choice = "tool_choice" in request_data
            logger.debug(
                "converting_messages_to_response_api",
                format_type=format_type,
                has_tools=has_tools,
                has_tool_choice=has_tool_choice,
            )
            response_request = self._response_adapter.chat_to_response_request(
                request_data
            )
            codex_request = response_request.model_dump()

            # Ensure Codex-specific defaults
            if "model" not in codex_request:
                codex_request["model"] = "gpt-5"

            logger.debug(
                "codex_request_conversion",
                format_type=format_type,
                original_keys=list(request_data.keys()),
                converted_keys=list(codex_request.keys()),
                tools_count=len(codex_request.get("tools", [])),
                tool_choice=codex_request.get("tool_choice", "auto"),
            )
            return codex_request

        # Native Response API format - passthrough
        logger.debug("request_passthrough", request_keys=list(request_data.keys()))
        return request_data

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to Messages format.
        
        Converts to OpenAI Chat Completions format, which is compatible
        with Anthropic Messages format for most use cases.

        Args:
            response_data: Codex Response API response

        Returns:
            OpenAI Chat Completions formatted response (compatible with Messages)
        """
        # Check if this is a Response API format response
        if self._is_response_api_format(response_data):
            # Check for tool calls in response
            has_tool_calls = self._has_tool_calls_in_response(response_data)
            logger.debug(
                "converting_response_api_to_chat_completions",
                has_tool_calls=has_tool_calls,
            )
            chat_response = self._response_adapter.response_to_chat_completion(
                response_data
            )
            converted_response = chat_response.model_dump()
            
            # Log conversion details
            choices = converted_response.get("choices", [])
            if choices:
                choice = choices[0]
                message = choice.get("message", {})
                tool_calls = message.get("tool_calls", [])
                logger.debug(
                    "response_conversion_completed",
                    finish_reason=choice.get("finish_reason"),
                    tool_calls_count=len(tool_calls),
                    content_length=len(message.get("content", "") or ""),
                )
            
            return converted_response

        return response_data

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert individual Response API events to Chat Completions format.

        Uses the same pattern as OpenAI adapter's streaming processor.

        Args:
            stream: Individual Response API events (already parsed by provider handler)

        Yields:
            OpenAI Chat Completions streaming chunks
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of stream adaptation."""
        # Extract the stream processing logic from OpenAI adapter pattern
        import time
        from collections import defaultdict

        from ccproxy.adapters.openai.models import generate_openai_response_id

        message_id = generate_openai_response_id()
        created = int(time.time())
        role_sent = False
        event_counts: dict[str, int] = defaultdict(int)
        start_time = time.time()

        logger.info(
            "streaming_started",
            plugin="codex",
            message_id=message_id,
        )

        async for event in stream:
            event_type = event.get("type", "unknown")
            event_counts[event_type] += 1

            # Log at TRACE level for each event
            logger.trace(
                "stream_event",
                event_type=event_type,
                message_id=message_id,
            )

            # Send initial role chunk if not sent yet
            if not role_sent:
                yield {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                role_sent = True

            # Convert Response API events to ChatCompletion deltas
            chunk = self._convert_response_event_to_chat_delta(
                event, message_id, created
            )
            if chunk:
                logger.trace(
                    "yielding_chat_chunk",
                    message_id=message_id,
                )
                yield chunk

        # Log streaming summary
        duration = time.time() - start_time
        logger.info(
            "streaming_complete",
            plugin="codex",
            message_id=message_id,
            duration_ms=round(duration * 1000, 2),
            event_summary=dict(event_counts),
            total_events=sum(event_counts.values()),
        )

    def _convert_response_event_to_chat_delta(
        self, event: dict[str, Any], stream_id: str, created: int
    ) -> dict[str, Any] | None:
        """Convert a Response API event to ChatCompletion delta format."""
        event_type = event.get("type", "")

        # Handle content deltas (main text output)
        if event_type == "response.output_text.delta":
            delta_text = event.get("delta", "")
            if delta_text:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle structured output deltas
        elif event_type == "response.output.delta":
            # Extract content from nested output structure
            output = event.get("output", [])
            delta_content = ""

            for output_item in output:
                if output_item.get("type") == "message":
                    content_blocks = output_item.get("content", [])
                    for block in content_blocks:
                        if block.get("type") in ["output_text", "text"]:
                            delta_content += block.get("text", "")

            if delta_content:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_content},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle completion events
        elif event_type == "response.completed":
            response = event.get("response", {})
            usage = response.get("usage")

            chunk_data = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model", "gpt-5"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            # Add usage if available
            if usage:
                chunk_data["usage"] = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            return chunk_data

        # Skip other event types - log at TRACE level to reduce noise
        if hasattr(logger, "trace"):
            logger.trace(
                "skipping_event_type", event_type=event_type, category="streaming"
            )
        return None

    def _is_response_api_format(self, response_data: dict[str, Any]) -> bool:
        """Check if response is in Response API format (used by Codex)."""
        # Response API responses have 'output' field or are wrapped in 'response'
        return "output" in response_data or "response" in response_data
    
    def _has_tool_calls_in_response(self, response_data: dict[str, Any]) -> bool:
        """Check if response contains tool calls."""
        # Check direct response format
        output = response_data.get("output", [])
        if output:
            for output_item in output:
                if output_item.get("type") == "message":
                    content = output_item.get("content", [])
                    for block in content:
                        if block.get("type") == "tool_call":
                            return True
        
        # Check wrapped response format
        response = response_data.get("response", {})
        if response:
            return self._has_tool_calls_in_response(response)
            
        return False
    
    def _detect_request_format(self, request_data: dict[str, Any]) -> str:
        """Detect whether request is OpenAI Chat Completions or Anthropic Messages format.
        
        Args:
            request_data: The request data
            
        Returns:
            Format type: 'openai_chat' or 'anthropic_messages'
        """
        # Check for Anthropic-specific fields
        if "max_tokens" in request_data:
            return "anthropic_messages"
        
        # Check for OpenAI-specific fields
        if "max_completion_tokens" in request_data or "max_tokens" not in request_data:
            return "openai_chat"
            
        # Default to openai_chat if unclear
        return "openai_chat"
