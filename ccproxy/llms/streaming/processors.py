"""Stream processing utilities for converting between different streaming formats.

This module provides stream processors that convert between different LLM
streaming response formats (e.g., Anthropic to OpenAI, OpenAI to Anthropic).
"""

import json
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from ccproxy.core.logging import get_logger

from .formatters import AnthropicSSEFormatter, OpenAISSEFormatter


logger = get_logger(__name__)


class AnthropicStreamProcessor:
    """Processes OpenAI streaming data into Anthropic SSE format."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize the stream processor.

        Args:
            model: Model name for responses
        """
        self.model = model
        self.formatter = AnthropicSSEFormatter()

    async def process_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[str]:
        """Process OpenAI-format streaming data into Anthropic SSE format.

        Args:
            stream: Async iterator of OpenAI-style response chunks

        Yields:
            Anthropic-formatted SSE strings with proper event: lines
        """
        message_started = False
        content_block_started = False

        async for chunk in stream:
            if not isinstance(chunk, dict):
                continue

            chunk_type = chunk.get("type")

            if chunk_type == "message_start":
                if not message_started:
                    yield self.formatter.format_event("message_start", chunk)
                    message_started = True

            elif chunk_type == "content_block_start":
                if not content_block_started:
                    yield self.formatter.format_event("content_block_start", chunk)
                    content_block_started = True

            elif chunk_type == "content_block_delta":
                yield self.formatter.format_event("content_block_delta", chunk)

            elif chunk_type == "ping":
                yield self.formatter.format_ping()

            elif chunk_type == "content_block_stop":
                yield self.formatter.format_event("content_block_stop", chunk)

            elif chunk_type == "message_delta":
                yield self.formatter.format_event("message_delta", chunk)

            elif chunk_type == "message_stop":
                yield self.formatter.format_event("message_stop", chunk)
                break


class OpenAIStreamProcessor:
    """Processes Anthropic/Claude streaming responses into OpenAI format."""

    def __init__(
        self,
        message_id: str | None = None,
        model: str = "claude-3-5-sonnet-20241022",
        created: int | None = None,
        enable_usage: bool = True,
        enable_tool_calls: bool = True,
        enable_thinking_serialization: bool | None = None,
        output_format: Literal["sse", "dict"] = "sse",
    ):
        """Initialize the stream processor.

        Args:
            message_id: Response ID, generated if not provided
            model: Model name for responses
            created: Creation timestamp, current time if not provided
            enable_usage: Whether to include usage information
            enable_tool_calls: Whether to process tool calls
            output_format: Output format - "sse" for Server-Sent Events strings, "dict" for dict objects
        """
        # Import here to avoid circular imports
        from ccproxy.llms.models.openai import generate_responses_id

        self.message_id = message_id or generate_responses_id()
        self.model = model
        self.created = created or int(time.time())
        self.enable_usage = enable_usage
        self.enable_tool_calls = enable_tool_calls
        self.output_format = output_format
        if enable_thinking_serialization is None:
            # Prefer service Settings.llm.openai_thinking_xml if available
            setting_val: bool | None = None
            try:
                from ccproxy.config.settings import Settings

                cfg = Settings.from_config()
                setting_val = bool(
                    getattr(getattr(cfg, "llm", {}), "openai_thinking_xml", True)
                )
            except Exception:
                setting_val = None

            if setting_val is not None:
                self.enable_thinking_serialization = setting_val
            else:
                # Fallback to env-based toggle
                env_val = (
                    os.getenv("LLM__OPENAI_THINKING_XML")
                    or os.getenv("OPENAI_STREAM_ENABLE_THINKING_SERIALIZATION")
                    or "true"
                ).lower()
                self.enable_thinking_serialization = env_val not in (
                    "0",
                    "false",
                    "no",
                    "off",
                )
        else:
            self.enable_thinking_serialization = enable_thinking_serialization
        self.formatter = OpenAISSEFormatter()

        # State tracking
        self.role_sent = False
        self.accumulated_content = ""
        self.tool_calls: dict[str, dict[str, Any]] = {}
        self.usage_info: dict[str, int] | None = None
        # Thinking block tracking
        self.current_thinking_text = ""
        self.current_thinking_signature: str | None = None
        self.thinking_block_active = False

    async def process_stream(
        self, claude_stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Process a Claude/Anthropic stream into OpenAI format.

        Args:
            claude_stream: Async iterator of Claude response chunks

        Yields:
            OpenAI-formatted SSE strings or dict objects based on output_format
        """
        # Get logger with request context at the start of the function
        logger = get_logger(__name__)

        try:
            chunk_count = 0
            processed_count = 0
            logger.debug(
                "openai_stream_processor_start",
                message_id=self.message_id,
                model=self.model,
                output_format=self.output_format,
                enable_usage=self.enable_usage,
                enable_tool_calls=self.enable_tool_calls,
                category="streaming_conversion",
                enable_thinking_serialization=self.enable_thinking_serialization,
            )

            async for chunk in claude_stream:
                chunk_count += 1
                chunk_type = chunk.get("type", "unknown")

                logger.trace(
                    "openai_processor_input_chunk",
                    chunk_number=chunk_count,
                    chunk_type=chunk_type,
                    category="format_detection",
                )

                async for sse_chunk in self._process_chunk(chunk):
                    processed_count += 1
                    yield sse_chunk

            logger.debug(
                "openai_stream_complete",
                total_chunks=chunk_count,
                processed_chunks=processed_count,
                message_id=self.message_id,
                category="streaming_conversion",
            )

            # Send final chunk
            if self.usage_info and self.enable_usage:
                yield self._format_chunk_output(
                    finish_reason="stop",
                    usage=self.usage_info,
                )
            else:
                yield self._format_chunk_output(finish_reason="stop")

            # Send DONE event (only for SSE format)
            if self.output_format == "sse":
                yield self.formatter.format_done()

        except (OSError, PermissionError) as e:
            logger.error("stream_processing_io_error", error=str(e), exc_info=e)
            # Send error chunk for IO errors
            if self.output_format == "sse":
                yield self.formatter.format_error_chunk(
                    self.message_id,
                    self.model,
                    self.created,
                    "error",
                    f"IO error: {str(e)}",
                )
                yield self.formatter.format_done()
            else:
                # Dict format error
                yield self._create_chunk_dict(finish_reason="error")
        except Exception as e:
            logger.error("stream_processing_error", error=str(e), exc_info=e)
            # Send error chunk
            if self.output_format == "sse":
                yield self.formatter.format_error_chunk(
                    self.message_id, self.model, self.created, "error", str(e)
                )
                yield self.formatter.format_done()
            else:
                # Dict format error
                yield self._create_chunk_dict(finish_reason="error")

    async def _process_chunk(
        self, chunk: dict[str, Any]
    ) -> AsyncIterator[str | dict[str, Any]]:
        """Process a single chunk from the Claude stream.

        Args:
            chunk: Claude response chunk

        Yields:
            OpenAI-formatted SSE strings or dict objects based on output_format
        """
        # Handle both Claude SDK and standard Anthropic API formats:
        # Claude SDK format: {"event": "...", "data": {"type": "..."}}
        # Anthropic API format: {"type": "...", ...}
        event_type = chunk.get("event")
        if event_type:
            # Claude SDK format
            chunk_data = chunk.get("data", {})
            chunk_type = chunk_data.get("type")
            format_source = "claude_sdk"
        else:
            # Standard Anthropic API format
            chunk_data = chunk
            chunk_type = chunk.get("type")
            format_source = "anthropic_api"

        logger.trace(
            "openai_processor_chunk_conversion",
            format_source=format_source,
            chunk_type=chunk_type,
            event_type=event_type,
            category="format_detection",
        )

        if chunk_type == "message_start":
            # Send initial role chunk
            if not self.role_sent:
                logger.trace(
                    "openai_conversion_message_start",
                    action="sending_role_chunk",
                    category="streaming_conversion",
                )
                yield self._format_chunk_output(delta={"role": "assistant"})
                self.role_sent = True

        elif chunk_type == "content_block_start":
            block = chunk_data.get("content_block", {})
            if block.get("type") == "thinking":
                # Start of thinking block
                self.thinking_block_active = True
                self.current_thinking_text = ""
                self.current_thinking_signature = None
            elif block.get("type") == "system_message":
                # Handle system message content block
                system_text = block.get("text", "")
                source = block.get("source", "ccproxy")
                # Format as text with clear source attribution
                formatted_text = f"[{source}]: {system_text}"
                yield self._format_chunk_output(delta={"content": formatted_text})
            elif block.get("type") == "tool_use_sdk" and self.enable_tool_calls:
                # Handle custom tool_use_sdk content block
                tool_id = block.get("id", "")
                tool_name = block.get("name", "")
                tool_input = block.get("input", {})
                source = block.get("source", "ccproxy")

                # For dict format, immediately yield the tool call
                if self.output_format == "dict":
                    yield self._format_chunk_output(
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "arguments": json.dumps(tool_input),
                                    },
                                }
                            ]
                        }
                    )
                else:
                    # For SSE format, store for later processing
                    self.tool_calls[tool_id] = {
                        "id": tool_id,
                        "name": tool_name,
                        "arguments": tool_input,
                        "source": source,
                    }
            elif block.get("type") == "tool_result_sdk":
                # Handle custom tool_result_sdk content block
                source = block.get("source", "ccproxy")
                tool_use_id = block.get("tool_use_id", "")
                result_content = block.get("content", "")
                is_error = block.get("is_error", False)
                error_indicator = " (ERROR)" if is_error else ""
                formatted_text = f"[{source} tool_result {tool_use_id}{error_indicator}]: {result_content}"
                yield self._format_chunk_output(delta={"content": formatted_text})
            elif block.get("type") == "result_message":
                # Handle custom result_message content block
                source = block.get("source", "ccproxy")
                result_data = block.get("data", {})
                session_id = result_data.get("session_id", "")
                stop_reason = result_data.get("stop_reason", "")
                usage = result_data.get("usage", {})
                cost_usd = result_data.get("total_cost_usd")
                formatted_text = f"[{source} result {session_id}]: stop_reason={stop_reason}, usage={usage}"
                if cost_usd is not None:
                    formatted_text += f", cost_usd={cost_usd}"
                yield self._format_chunk_output(delta={"content": formatted_text})

            elif block.get("type") == "tool_use":
                # Start of tool call
                tool_id = block.get("id", "")
                tool_name = block.get("name", "")
                self.tool_calls[tool_id] = {
                    "id": tool_id,
                    "name": tool_name,
                    "arguments": "",
                }

        elif chunk_type == "content_block_delta":
            delta = chunk_data.get("delta", {})
            delta_type = delta.get("type")

            if delta_type == "text_delta":
                # Text content
                text = delta.get("text", "")
                if text:
                    logger.trace(
                        "openai_conversion_text_delta",
                        text_length=len(text),
                        category="streaming_conversion",
                    )
                    yield self._format_chunk_output(delta={"content": text})

            elif delta_type == "thinking_delta" and self.thinking_block_active:
                # Thinking content
                thinking_text = delta.get("thinking", "")
                if thinking_text:
                    self.current_thinking_text += thinking_text

            elif delta_type == "signature_delta" and self.thinking_block_active:
                # Thinking signature
                signature = delta.get("signature", "")
                if signature:
                    if self.current_thinking_signature is None:
                        self.current_thinking_signature = ""
                    self.current_thinking_signature += signature

            elif delta_type == "input_json_delta":
                # Tool call arguments
                partial_json = delta.get("partial_json", "")
                if partial_json and self.tool_calls:
                    # Find the tool call this belongs to (usually the last one)
                    latest_tool_id = list(self.tool_calls.keys())[-1]
                    self.tool_calls[latest_tool_id]["arguments"] += partial_json

        elif chunk_type == "content_block_stop":
            # End of content block
            if self.thinking_block_active:
                # Format and send the complete thinking block
                self.thinking_block_active = False
                if self.current_thinking_text:
                    # Format thinking block with signature
                    if self.enable_thinking_serialization:
                        thinking_content = (
                            f'<thinking signature="{self.current_thinking_signature}">'
                            f"{self.current_thinking_text}</thinking>"
                        )
                        yield self._format_chunk_output(
                            delta={"content": thinking_content}
                        )
                # Reset thinking state
                self.current_thinking_text = ""
                self.current_thinking_signature = None

            elif (
                self.tool_calls
                and self.enable_tool_calls
                and self.output_format == "sse"
            ):
                # Send completed tool calls (only for SSE format, dict format sends immediately)
                for tool_call in self.tool_calls.values():
                    yield self._format_chunk_output(
                        delta={
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": tool_call["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tool_call["name"],
                                        "arguments": json.dumps(tool_call["arguments"])
                                        if isinstance(tool_call["arguments"], dict)
                                        else tool_call["arguments"],
                                    },
                                }
                            ]
                        }
                    )

        elif chunk_type == "message_delta":
            # Usage information
            usage = chunk_data.get("usage", {})
            if usage and self.enable_usage:
                logger.trace(
                    "openai_conversion_usage_info",
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    category="streaming_conversion",
                )
                self.usage_info = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("input_tokens", 0)
                    + usage.get("output_tokens", 0),
                }

        elif chunk_type == "message_stop":
            # End of message - handled in main process_stream method
            pass

    def _create_chunk_dict(
        self,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Create an OpenAI completion chunk dict.

        Args:
            delta: The delta content for the chunk
            finish_reason: Optional finish reason
            usage: Optional usage information

        Returns:
            OpenAI completion chunk dict
        """
        chunk = {
            "id": self.message_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta or {},
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage:
            chunk["usage"] = usage

        return chunk

    def _format_chunk_output(
        self,
        delta: dict[str, Any] | None = None,
        finish_reason: str | None = None,
        usage: dict[str, int] | None = None,
    ) -> str | dict[str, Any]:
        """Format chunk output based on output_format flag.

        Args:
            delta: The delta content for the chunk
            finish_reason: Optional finish reason
            usage: Optional usage information

        Returns:
            Either SSE string or dict based on output_format
        """
        if self.output_format == "dict":
            return self._create_chunk_dict(delta, finish_reason, usage)
        else:
            # SSE format
            if finish_reason:
                if usage:
                    return self.formatter.format_final_chunk(
                        self.message_id,
                        self.model,
                        self.created,
                        finish_reason,
                        usage=usage,
                    )
                else:
                    return self.formatter.format_final_chunk(
                        self.message_id, self.model, self.created, finish_reason
                    )
            elif delta and delta.get("role"):
                return self.formatter.format_first_chunk(
                    self.message_id, self.model, self.created, delta["role"]
                )
            elif delta and delta.get("content"):
                return self.formatter.format_content_chunk(
                    self.message_id, self.model, self.created, delta["content"]
                )
            elif delta and delta.get("tool_calls"):
                # Handle tool calls
                tool_call = delta["tool_calls"][0]  # Assume single tool call for now
                return self.formatter.format_tool_call_chunk(
                    self.message_id,
                    self.model,
                    self.created,
                    tool_call["id"],
                    tool_call.get("function", {}).get("name"),
                    tool_call.get("function", {}).get("arguments"),
                )
            else:
                # Empty delta - send chunk with null finish_reason
                return self.formatter.format_content_chunk(
                    self.message_id, self.model, self.created, ""
                )
