"""Unified stream transformation framework.

This module provides a flexible framework for transforming various streaming formats
to OpenAI-compatible SSE (Server-Sent Events) format. It supports multiple input sources:

1. Claude SDK responses (dict chunks from the Claude Python SDK)
2. SSE responses (from reverse proxy to Anthropic API)

The framework is designed to be extensible for future format support.

Key components:
- StreamingConfig: Configuration for transformation behavior
- EventSource: Abstract interface for different event sources
- OpenAIStreamTransformer: Main transformer with factory methods for different sources

For format-specific utilities, see:
- openai_streaming_formatter.py: OpenAI SSE formatting utilities
- anthropic_streaming.py: Anthropic native SSE format utilities
"""

import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from ccproxy.services.openai_streaming_formatter import OpenAIStreamingFormatter
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


@dataclass
class StreamingConfig:
    """Configuration for streaming behavior."""

    enable_text_chunking: bool = True
    enable_tool_calls: bool = True
    enable_usage_info: bool = True
    chunk_delay_ms: float = 10.0
    chunk_size_words: int = 3


@dataclass
class StreamEvent:
    """Unified internal event format for streaming."""

    type: str
    data: dict[str, Any] = field(default_factory=dict)


class EventSource(ABC):
    """Abstract base class for event sources."""

    @abstractmethod
    def get_events(self) -> AsyncIterator[StreamEvent]:
        """Get events from the source."""
        raise NotImplementedError


class ClaudeSDKEventSource(EventSource):
    """Event source for Claude SDK responses."""

    def __init__(self, claude_response_iterator: AsyncGenerator[dict[str, Any], None]):
        self.iterator = claude_response_iterator

    def get_events(self) -> AsyncIterator[StreamEvent]:
        """Convert Claude SDK chunks to unified events."""
        return self._get_events_impl()

    async def _get_events_impl(self) -> AsyncIterator[StreamEvent]:
        """Implementation of get_events."""
        async for chunk in self.iterator:
            chunk_type = chunk.get("type", "")

            if chunk_type == "message_start":
                yield StreamEvent(
                    type="start", data={"message": chunk.get("message", {})}
                )

            elif chunk_type == "content_block_start":
                content_block = chunk.get("content_block", {})
                yield StreamEvent(
                    type="content_block_start", data={"block": content_block}
                )

            elif chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                yield StreamEvent(type="content_block_delta", data={"delta": delta})

            elif chunk_type == "content_block_stop":
                yield StreamEvent(type="content_block_stop", data={})

            elif chunk_type == "message_delta":
                delta = chunk.get("delta", {})
                usage = chunk.get("usage", {})
                yield StreamEvent(
                    type="message_delta", data={"delta": delta, "usage": usage}
                )

            elif chunk_type == "message_stop":
                yield StreamEvent(type="message_stop", data={})


class SSEEventSource(EventSource):
    """Event source for SSE (Server-Sent Events) responses."""

    def __init__(self, response: Any) -> None:
        """Initialize with httpx streaming response."""
        self.response = response

    def get_events(self) -> AsyncIterator[StreamEvent]:
        """Parse SSE stream and convert to unified events."""
        return self._get_events_impl()

    async def _get_events_impl(self) -> AsyncIterator[StreamEvent]:
        """Implementation of get_events."""
        buffer = b""

        async for chunk in self.response.aiter_bytes():
            buffer += chunk

            # Process complete lines
            while b"\n" in buffer:
                line_end = buffer.find(b"\n")
                line = buffer[:line_end]
                buffer = buffer[line_end + 1 :]

                # Skip empty lines
                if not line.strip():
                    continue

                line_str = line.decode("utf-8", errors="replace").strip()

                # Parse SSE format
                if line_str.startswith("data:"):
                    data_str = line_str[5:].strip()

                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", "")

                        if event_type == "message_start":
                            yield StreamEvent(
                                type="start", data={"message": data.get("message", {})}
                            )

                        elif event_type == "content_block_start":
                            content_block = data.get("content_block", {})
                            yield StreamEvent(
                                type="content_block_start",
                                data={"block": content_block},
                            )

                        elif event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            yield StreamEvent(
                                type="content_block_delta", data={"delta": delta}
                            )

                        elif event_type == "content_block_stop":
                            yield StreamEvent(type="content_block_stop", data={})

                        elif event_type == "message_delta":
                            delta = data.get("delta", {})
                            usage = data.get("usage", {})
                            yield StreamEvent(
                                type="message_delta",
                                data={"delta": delta, "usage": usage},
                            )

                        elif event_type == "message_stop":
                            yield StreamEvent(type="message_stop", data={})

                    except json.JSONDecodeError:
                        logger.debug(f"Failed to parse SSE data: {data_str}")
                    except Exception as e:
                        logger.error(
                            f"Error processing SSE event: {e}, data: {data_str[:100]}"
                        )


class OpenAIStreamTransformer:
    """Unified transformer for converting streams to OpenAI format."""

    def __init__(
        self,
        event_source: EventSource,
        message_id: str | None = None,
        model: str = "gpt-4o",
        created: int | None = None,
        config: StreamingConfig | None = None,
    ):
        """Initialize the transformer.

        Args:
            event_source: Source of events to transform
            message_id: Unique message identifier
            model: Model name
            created: Unix timestamp
            config: Streaming configuration
        """
        self.event_source = event_source
        self.message_id = message_id or f"chatcmpl-{uuid.uuid4().hex[:29]}"
        self.model = model
        self.created = created or int(time.time())
        self.config = config or StreamingConfig()
        self.formatter = OpenAIStreamingFormatter()

    @classmethod
    def from_claude_sdk(
        cls,
        claude_response_iterator: AsyncGenerator[dict[str, Any], None],
        message_id: str | None = None,
        model: str = "gpt-4o",
        created: int | None = None,
        config: StreamingConfig | None = None,
    ) -> "OpenAIStreamTransformer":
        """Create transformer for Claude SDK responses."""
        event_source = ClaudeSDKEventSource(claude_response_iterator)
        return cls(event_source, message_id, model, created, config)

    @classmethod
    def from_sse_stream(
        cls,
        response: Any,
        message_id: str | None = None,
        model: str = "gpt-4o",
        created: int | None = None,
        config: StreamingConfig | None = None,
    ) -> "OpenAIStreamTransformer":
        """Create transformer for SSE stream responses."""
        event_source = SSEEventSource(response)
        return cls(event_source, message_id, model, created, config)

    def _split_text_for_streaming(self, text: str) -> list[str]:
        """Split text into smaller chunks for better streaming experience."""
        if not self.config.enable_text_chunking:
            return [text]

        if not text or len(text) <= 10:
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
            if word_count >= self.config.chunk_size_words or "\n" in word:
                if current_chunk.strip():
                    chunks.append(current_chunk)
                current_chunk = ""
                word_count = 0

        # Add remaining text
        if current_chunk.strip():
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    async def transform(self) -> AsyncGenerator[str, None]:
        """Transform events to OpenAI SSE format.

        Yields:
            Formatted OpenAI SSE strings
        """
        has_sent_role = False
        has_content = False
        in_thinking_block = False
        tool_calls: dict[str, dict[str, Any]] = {}
        current_tool_index = 0

        try:
            async for event in self.event_source.get_events():
                logger.debug(f"Processing event: {event.type}")

                if event.type == "start":
                    # Send initial role chunk
                    if not has_sent_role:
                        yield self.formatter.format_first_chunk(
                            self.message_id, self.model, self.created
                        )
                        has_sent_role = True

                elif event.type == "content_block_start":
                    block = event.data.get("block", {})
                    block_type = block.get("type")

                    if block_type == "thinking" or (
                        block_type == "text" and block.get("thinking", False)
                    ):
                        # Start thinking block
                        in_thinking_block = True
                        has_content = True
                        yield self.formatter.format_content_chunk(
                            self.message_id, self.model, self.created, "[Thinking]\n"
                        )
                        logger.debug(f"Started thinking block (type: {block_type})")

                    elif block_type == "tool_use" and self.config.enable_tool_calls:
                        # Start tool call
                        tool_call_id = block.get("id", str(uuid.uuid4()))
                        function_name = block.get("name", "")
                        tool_calls[tool_call_id] = {
                            "id": tool_call_id,
                            "name": function_name,
                            "arguments": "",
                            "index": current_tool_index,
                        }
                        yield self.formatter.format_tool_call_chunk(
                            self.message_id,
                            self.model,
                            self.created,
                            tool_call_id,
                            function_name,
                            "",
                            current_tool_index,
                        )
                        current_tool_index += 1

                elif event.type == "content_block_delta":
                    delta = event.data.get("delta", {})
                    delta_type = delta.get("type")

                    if delta_type == "thinking_delta":
                        # Handle thinking content
                        thinking_text = delta.get("thinking", "")
                        if thinking_text:
                            has_content = True
                            yield self.formatter.format_content_chunk(
                                self.message_id, self.model, self.created, thinking_text
                            )

                    elif delta_type == "text_delta":
                        # End thinking block if needed
                        if in_thinking_block:
                            in_thinking_block = False
                            yield self.formatter.format_content_chunk(
                                self.message_id, self.model, self.created, "\n---\n"
                            )

                        # Handle regular text
                        text = delta.get("text", "")
                        if text:
                            has_content = True

                            if self.config.enable_text_chunking:
                                # Split and stream with delays
                                text_parts = self._split_text_for_streaming(text)
                                for i, part in enumerate(text_parts):
                                    yield self.formatter.format_content_chunk(
                                        self.message_id, self.model, self.created, part
                                    )
                                    if i < len(text_parts) - 1:
                                        await asyncio.sleep(
                                            self.config.chunk_delay_ms / 1000
                                        )
                            else:
                                # Stream as-is
                                yield self.formatter.format_content_chunk(
                                    self.message_id, self.model, self.created, text
                                )

                    elif (
                        delta_type == "input_json_delta"
                        and self.config.enable_tool_calls
                    ):
                        # Handle tool input
                        if tool_calls:
                            partial_json = delta.get("partial_json", "")
                            # Find the last tool call
                            tool_call_id = list(tool_calls.keys())[-1]
                            tool_call_data = tool_calls[tool_call_id]
                            tool_call_data["arguments"] += partial_json

                            yield self.formatter.format_tool_call_chunk(
                                self.message_id,
                                self.model,
                                self.created,
                                tool_call_id,
                                None,
                                partial_json,
                                tool_call_data["index"],
                            )

                elif event.type == "content_block_stop":
                    # End thinking block if still active
                    if in_thinking_block:
                        in_thinking_block = False
                        yield self.formatter.format_content_chunk(
                            self.message_id, self.model, self.created, "\n---\n"
                        )

                elif event.type == "message_delta":
                    # Handle message ending
                    delta = event.data.get("delta", {})
                    stop_reason = delta.get("stop_reason", "stop")

                    # Map stop reasons
                    finish_reason_map = {
                        "end_turn": "stop",
                        "max_tokens": "length",
                        "tool_use": "tool_calls",
                        "stop_sequence": "stop",
                        "pause_turn": "stop",
                        "refusal": "content_filter",
                    }
                    finish_reason = finish_reason_map.get(stop_reason, "stop")

                    # Include usage if enabled
                    usage_data = None
                    if self.config.enable_usage_info and "usage" in event.data:
                        usage_info = event.data.get("usage", {})
                        usage_data = {
                            "prompt_tokens": usage_info.get("input_tokens", 0),
                            "completion_tokens": usage_info.get("output_tokens", 0),
                            "total_tokens": (
                                usage_info.get("input_tokens", 0)
                                + usage_info.get("output_tokens", 0)
                            ),
                        }

                    yield self.formatter.format_final_chunk(
                        self.message_id,
                        self.model,
                        self.created,
                        finish_reason,
                        usage=usage_data,
                    )
                    break

            # Send final chunk if no content
            if has_sent_role and not has_content:
                yield self.formatter.format_final_chunk(
                    self.message_id, self.model, self.created
                )

        except asyncio.CancelledError:
            logger.info("Stream transformation cancelled")
            yield self.formatter.format_final_chunk(
                self.message_id, self.model, self.created, "cancelled"
            )
            raise
        except Exception as e:
            logger.error(f"Error in stream transformation: {e}")
            yield self.formatter.format_error_chunk(
                self.message_id, self.model, self.created, "internal_error", str(e)
            )
        finally:
            # Always send DONE
            yield self.formatter.format_done()


class AnthropicStreamTransformer:
    """Unified transformer for converting streams to Anthropic format."""

    def __init__(
        self,
        event_source: EventSource,
        message_id: str,
        model: str,
    ):
        """Initialize the transformer.

        Args:
            event_source: Source of events to transform
            message_id: Unique message identifier
            model: Model name
        """
        self.event_source = event_source
        self.message_id = message_id
        self.model = model
        self.formatter = self._get_formatter()

    def _get_formatter(self) -> Any:
        """Get the Anthropic formatter."""
        from ccproxy.services.anthropic_streaming import StreamingFormatter

        return StreamingFormatter()

    @classmethod
    def from_claude_sdk(
        cls,
        claude_response_iterator: AsyncGenerator[dict[str, Any], None],
        message_id: str,
        model: str,
    ) -> "AnthropicStreamTransformer":
        """Create transformer for Claude SDK responses."""
        event_source = ClaudeSDKEventSource(claude_response_iterator)
        return cls(event_source, message_id, model)

    @classmethod
    def from_sse_stream(
        cls,
        response: Any,
        message_id: str,
        model: str,
    ) -> "AnthropicStreamTransformer":
        """Create transformer for SSE stream responses."""
        event_source = SSEEventSource(response)
        return cls(event_source, message_id, model)

    async def transform(self) -> AsyncGenerator[str, None]:
        """Transform events to Anthropic SSE format.

        Yields:
            Formatted Anthropic SSE strings
        """
        try:
            # Send message start event
            yield self.formatter.format_message_start(self.message_id, self.model)

            # Send content block start
            yield self.formatter.format_content_block_start()

            # Process events
            has_content = False
            async for event in self.event_source.get_events():
                logger.debug(f"Processing event: {event.type}")

                if event.type == "content_block_delta":
                    delta = event.data.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            has_content = True
                            yield self.formatter.format_content_block_delta(text)

                elif event.type == "content_block_stop":
                    yield self.formatter.format_content_block_stop()

                elif event.type == "message_delta":
                    # Content block must be stopped before message delta
                    if has_content:
                        yield self.formatter.format_content_block_stop()
                    delta = event.data.get("delta", {})
                    stop_reason = delta.get("stop_reason", "end_turn")
                    stop_sequence = delta.get("stop_sequence")
                    yield self.formatter.format_message_delta(
                        stop_reason, stop_sequence
                    )
                    # Claude SDK doesn't send message_stop, so we need to synthesize it
                    yield self.formatter.format_message_stop()
                    break

                elif event.type == "message_stop":
                    yield self.formatter.format_message_stop()
                    break

            # If we never got content, still need to close properly
            if not has_content:
                yield self.formatter.format_content_block_stop()
                yield self.formatter.format_message_delta()
                yield self.formatter.format_message_stop()

        except asyncio.CancelledError:
            logger.info("Stream transformation cancelled")
            if not has_content:
                yield self.formatter.format_content_block_stop()
            yield self.formatter.format_message_delta(stop_reason="cancelled")
            yield self.formatter.format_message_stop()
            raise
        except Exception as e:
            logger.error(f"Error in stream transformation: {e}")
            # Close content block if it was started
            if not has_content:
                yield self.formatter.format_content_block_stop()
            yield self.formatter.format_message_delta(stop_reason="error")
            yield self.formatter.format_message_stop()
            # Format error and continue (don't re-raise for Anthropic compatibility)
            yield self.formatter.format_error("internal_server_error", str(e))
        finally:
            # Always send DONE at the end
            yield self.formatter.format_done()
