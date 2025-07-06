"""OpenAI-compatible streaming response utilities."""

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from claude_code_proxy.models.errors import ErrorDetail, StreamingError
from claude_code_proxy.utils.logging import get_logger


logger = get_logger(__name__)


def _split_text_for_streaming(text: str, chunk_size: int = 3) -> list[str]:
    """
    Split text into smaller chunks for better streaming experience.

    This simulates OpenAI's token-by-token streaming by breaking large
    text chunks into smaller pieces based on word boundaries.

    Args:
        text: Text to split
        chunk_size: Approximate number of words per chunk

    Returns:
        List of text chunks
    """
    if not text or len(text) <= 10:
        # Return small text as-is
        return [text]

    # Split by words but keep whitespace
    words = []
    current_word = ""

    for char in text:
        if char.isspace():
            if current_word:
                words.append(current_word)
                current_word = ""
            words.append(char)
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    # Group words into chunks
    chunks = []
    current_chunk = ""
    word_count = 0

    for word in words:
        current_chunk += word

        if not word.isspace():
            word_count += 1

        # Create chunk when we hit word limit or encounter newlines
        if word_count >= chunk_size or "\n" in word:
            if current_chunk.strip():
                chunks.append(current_chunk)
            current_chunk = ""
            word_count = 0

    # Add remaining text
    if current_chunk.strip():
        chunks.append(current_chunk)

    return chunks if chunks else [text]


class OpenAIStreamingFormatter:
    """Formats streaming responses to match OpenAI's SSE format."""

    @staticmethod
    def format_data_event(data: dict[str, Any]) -> str:
        """
        Format a data event for OpenAI-compatible Server-Sent Events.

        Args:
            data: Event data dictionary

        Returns:
            Formatted SSE string
        """
        json_data = json.dumps(data, separators=(",", ":"))
        return f"data: {json_data}\n\n"

    @staticmethod
    def format_first_chunk(
        message_id: str, model: str, created: int, role: str = "assistant"
    ) -> str:
        """
        Format the first chunk with role and basic metadata.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            role: Role of the assistant

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": role},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_content_chunk(
        message_id: str, model: str, created: int, content: str, choice_index: int = 0
    ) -> str:
        """
        Format a content chunk with text delta.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            content: Text content to include in the delta
            choice_index: Index of the choice (usually 0)

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"content": content},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_tool_call_chunk(
        message_id: str,
        model: str,
        created: int,
        tool_call_id: str,
        function_name: str | None = None,
        function_arguments: str | None = None,
        tool_call_index: int = 0,
        choice_index: int = 0,
    ) -> str:
        """
        Format a tool call chunk.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            tool_call_id: ID of the tool call
            function_name: Name of the function being called
            function_arguments: Arguments for the function
            tool_call_index: Index of the tool call
            choice_index: Index of the choice (usually 0)

        Returns:
            Formatted SSE string
        """
        tool_call: dict[str, Any] = {
            "index": tool_call_index,
            "id": tool_call_id,
            "type": "function",
            "function": {},
        }

        if function_name is not None:
            tool_call["function"]["name"] = function_name

        if function_arguments is not None:
            tool_call["function"]["arguments"] = function_arguments

        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tool_call]},
                    "logprobs": None,
                    "finish_reason": None,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_final_chunk(
        message_id: str,
        model: str,
        created: int,
        finish_reason: str = "stop",
        choice_index: int = 0,
    ) -> str:
        """
        Format the final chunk with finish_reason.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            finish_reason: Reason for completion (stop, length, tool_calls, etc.)
            choice_index: Index of the choice (usually 0)

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": choice_index,
                    "delta": {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_error_chunk(
        message_id: str, model: str, created: int, error_type: str, error_message: str
    ) -> str:
        """
        Format an error chunk.

        Args:
            message_id: Unique identifier for the completion
            model: Model name being used
            created: Unix timestamp when the completion was created
            error_type: Type of error
            error_message: Error message

        Returns:
            Formatted SSE string
        """
        data = {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": {}, "logprobs": None, "finish_reason": "error"}
            ],
            "error": {"type": error_type, "message": error_message},
        }
        return OpenAIStreamingFormatter.format_data_event(data)

    @staticmethod
    def format_done() -> str:
        """
        Format the final DONE event.

        Returns:
            Formatted SSE termination string
        """
        return "data: [DONE]\n\n"


async def stream_claude_response_openai(
    claude_response_iterator: AsyncGenerator[dict[str, Any], None],
    message_id: str,
    model: str,
    created: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Convert Claude SDK response to OpenAI-compatible streaming format.

    Args:
        claude_response_iterator: Async iterator of Claude response chunks
        message_id: Unique message identifier
        model: Model name being used
        created: Unix timestamp when the completion was created

    Yields:
        Formatted OpenAI-compatible SSE strings
    """
    import time

    if created is None:
        created = int(time.time())

    formatter = OpenAIStreamingFormatter()

    try:
        logger.debug(
            f"Starting OpenAI streaming conversion for message_id: {message_id}, model: {model}"
        )

        # Send first chunk with role
        first_chunk = formatter.format_first_chunk(message_id, model, created)
        logger.debug(f"Sending first chunk: {first_chunk[:100]}...")
        yield first_chunk

        # Process Claude response chunks
        has_content = False
        tool_calls: dict[str, dict[str, Any]] = {}
        chunk_counter = 0

        try:
            async for chunk in claude_response_iterator:
                chunk_counter += 1
                chunk_type = chunk.get("type")
                logger.debug(
                    f"Processing Claude chunk {chunk_counter} (type: {chunk_type}): {chunk}"
                )

                if chunk_type == "content_block_delta":
                    # Handle text content
                    text = chunk.get("delta", {}).get("text", "")
                    if text:
                        has_content = True

                        # Split large text chunks for better streaming experience
                        # This makes the streaming feel more like OpenAI's token-by-token streaming
                        text_parts = _split_text_for_streaming(text)

                        for i, part in enumerate(text_parts):
                            content_chunk = formatter.format_content_chunk(
                                message_id, model, created, part
                            )
                            logger.debug(
                                f"Sending content chunk: {repr(part)} -> {content_chunk[:100]}..."
                            )
                            yield content_chunk

                            # Add small delay between chunks for more natural streaming
                            if i < len(text_parts) - 1:  # Don't delay after last chunk
                                await asyncio.sleep(0.01)  # 10ms delay

                elif chunk_type == "content_block_start":
                    # Handle tool use start
                    content_block = chunk.get("content_block", {})
                    if content_block.get("type") == "tool_use":
                        tool_call_id = content_block.get("id", str(uuid.uuid4()))
                        function_name = content_block.get("name", "")
                        tool_calls[tool_call_id] = {
                            "id": tool_call_id,
                            "name": function_name,
                            "arguments": "",
                        }
                        yield formatter.format_tool_call_chunk(
                            message_id, model, created, tool_call_id, function_name, ""
                        )

                elif chunk_type == "content_block_delta":
                    # Handle tool use input delta
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "input_json_delta":
                        # This is tool arguments
                        partial_json = delta.get("partial_json", "")
                        if partial_json and tool_calls:
                            # Find the tool call this belongs to (typically the last one)
                            tool_call_id = list(tool_calls.keys())[-1]
                            tool_call_data = tool_calls[tool_call_id]
                            tool_call_data["arguments"] += partial_json
                            yield formatter.format_tool_call_chunk(
                                message_id,
                                model,
                                created,
                                tool_call_id,
                                None,
                                partial_json,
                            )

                elif chunk_type == "message_delta":
                    # Message is ending
                    delta = chunk.get("delta", {})
                    stop_reason = delta.get("stop_reason", "stop")

                    # Map Claude stop reasons to OpenAI finish reasons
                    finish_reason_map = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                        "stop_sequence": "stop",
                    }
                    finish_reason = finish_reason_map.get(stop_reason, "stop")

                    yield formatter.format_final_chunk(
                        message_id, model, created, finish_reason
                    )
                    break

            # If we never got content or tool calls, still need to send final chunk
            if not has_content and not tool_calls:
                logger.debug("No content received, sending empty final chunk")
                final_chunk = formatter.format_final_chunk(message_id, model, created)
                logger.debug(f"Sending empty final chunk: {final_chunk[:100]}...")
                yield final_chunk

        except asyncio.CancelledError:
            # Handle stream cancellation gracefully
            logger.info("OpenAI streaming response cancelled")
            yield formatter.format_final_chunk(message_id, model, created, "cancelled")
            raise

    except asyncio.CancelledError:
        # Handle outer cancellation
        logger.info("OpenAI streaming response cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI streaming response: {e}")
        yield formatter.format_error_chunk(
            message_id, model, created, "internal_server_error", str(e)
        )

    finally:
        # Always send DONE at the end
        done_chunk = formatter.format_done()
        logger.debug(f"Sending DONE: {done_chunk}")
        yield done_chunk


async def stream_claude_response_openai_simple(
    claude_response_iterator: AsyncGenerator[dict[str, Any], None],
    message_id: str,
    model: str,
    created: int | None = None,
) -> AsyncGenerator[str, None]:
    """
    Convert Claude SDK response to OpenAI-compatible streaming format (simplified).

    This is a simplified version that focuses on text-only responses without
    tool calling support for basic compatibility.

    Args:
        claude_response_iterator: Async iterator of Claude response chunks
        message_id: Unique message identifier
        model: Model name being used
        created: Unix timestamp when the completion was created

    Yields:
        Formatted OpenAI-compatible SSE strings
    """
    import time

    if created is None:
        created = int(time.time())

    formatter = OpenAIStreamingFormatter()

    try:
        # Send first chunk with role
        yield formatter.format_first_chunk(message_id, model, created)

        # Process Claude response chunks
        has_content = False

        try:
            async for chunk in claude_response_iterator:
                chunk_type = chunk.get("type")

                if chunk_type == "content_block_delta":
                    # Handle text content
                    text = chunk.get("delta", {}).get("text", "")
                    if text:
                        has_content = True
                        yield formatter.format_content_chunk(
                            message_id, model, created, text
                        )

                elif chunk_type == "message_delta":
                    # Message is ending
                    delta = chunk.get("delta", {})
                    stop_reason = delta.get("stop_reason", "stop")

                    # Map Claude stop reasons to OpenAI finish reasons
                    finish_reason_map = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                        "stop_sequence": "stop",
                    }
                    finish_reason = finish_reason_map.get(stop_reason, "stop")

                    yield formatter.format_final_chunk(
                        message_id, model, created, finish_reason
                    )
                    break

            # If we never got content, still need to send final chunk
            if not has_content:
                yield formatter.format_final_chunk(message_id, model, created)

        except asyncio.CancelledError:
            # Handle stream cancellation gracefully
            logger.info("OpenAI simple streaming response cancelled")
            yield formatter.format_final_chunk(message_id, model, created, "cancelled")
            raise

    except asyncio.CancelledError:
        # Handle outer cancellation
        logger.info("OpenAI simple streaming response cancelled")
        raise
    except Exception as e:
        logger.error(f"Error in OpenAI streaming response: {e}")
        yield formatter.format_error_chunk(
            message_id, model, created, "internal_server_error", str(e)
        )

    finally:
        # Always send DONE at the end
        done_chunk = formatter.format_done()
        logger.debug(f"Sending DONE: {done_chunk}")
        yield done_chunk
