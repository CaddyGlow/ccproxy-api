"""Anthropic to OpenAI format adapter - unidirectional conversion."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.adapters.base import APIAdapter
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


class AnthropicToOpenAIAdapter(APIAdapter):
    """Converts Anthropic format responses to OpenAI format - unidirectional."""

    def __init__(self) -> None:
        """Initialize the Anthropic to OpenAI adapter."""
        pass

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic request format to OpenAI format.

        Args:
            request: Anthropic format request

        Returns:
            OpenAI format request

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        logger = get_logger(__name__)

        try:
            # Build base OpenAI request
            openai_request: dict[str, Any] = {}

            # Direct field mappings
            if "model" in request:
                openai_request["model"] = request["model"]

            # Convert messages, handling system field
            messages = self._convert_anthropic_messages_to_openai(request)
            openai_request["messages"] = messages

            # Handle optional parameters
            self._handle_optional_parameters(request, openai_request)

            # Handle tools and tool choice
            self._handle_tools(request, openai_request)

            # Handle metadata
            self._handle_metadata(request, openai_request)

            logger.debug(
                "anthropic_to_openai_conversion_completed",
                from_format="anthropic",
                to_format="openai",
                original_model=request.get("model"),
                openai_model=openai_request.get("model"),
                has_tools=bool(openai_request.get("tools")),
                message_count=len(openai_request.get("messages", [])),
                operation="adapt_request",
            )

            return openai_request

        except Exception as e:
            logger.error(
                "anthropic_to_openai_conversion_failed",
                error=str(e),
                request_keys=list(request.keys())
                if isinstance(request, dict)
                else "not_dict",
                operation="adapt_request",
                exc_info=e,
            )
            raise ValueError(f"Invalid Anthropic request format: {e}") from e

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI response format to Anthropic format.

        Args:
            response: OpenAI format response

        Returns:
            Anthropic format response

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        logger = get_logger(__name__)

        # Check if this is an error response
        if "error" in response:
            logger.debug(
                "anthropic_to_openai_adapter_error_response_detected",
                error_type=response.get("error", {}).get("type"),
                error_message=response.get("error", {}).get("message", ""),
            )
            return self.adapt_error(response)

        try:
            # Extract original model from response
            model = response.get("model", "gpt-4")

            # Create Anthropic response base
            anthropic_response: dict[str, Any] = {
                "type": "message",
                "role": "assistant",
                "model": model,
            }

            # Add id if present
            if "id" in response:
                anthropic_response["id"] = response["id"]

            # Convert choices to content blocks
            content_blocks, stop_reason = self._convert_openai_choices_to_content(
                response
            )
            anthropic_response["content"] = content_blocks
            anthropic_response["stop_reason"] = stop_reason

            # Convert usage information
            if "usage" in response:
                anthropic_response["usage"] = self._convert_openai_usage(
                    response["usage"]
                )

            logger.debug(
                "openai_to_anthropic_conversion_completed",
                from_format="openai",
                to_format="anthropic",
                original_model=model,
                content_blocks=len(content_blocks),
                stop_reason=stop_reason,
                operation="adapt_response",
            )

            return anthropic_response

        except Exception as e:
            logger.error(
                "openai_to_anthropic_conversion_failed",
                error=str(e),
                response_keys=list(response.keys())
                if isinstance(response, dict)
                else "not_dict",
                operation="adapt_response",
                exc_info=e,
            )
            raise ValueError(f"Invalid OpenAI response format: {e}") from e

    def adapt_error(self, error_response: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI error format to Anthropic error format.

        Args:
            error_response: OpenAI error response

        Returns:
            Anthropic error response
        """
        # Extract error details from OpenAI format
        openai_error = error_response.get("error", {})
        error_type = openai_error.get("type", "internal_server_error")
        error_message = openai_error.get("message", "An error occurred")

        # Map OpenAI error types to Anthropic error types
        error_type_mapping = {
            "invalid_request_error": "invalid_request_error",
            "authentication_error": "authentication_error",
            "permission_error": "permission_error",
            "not_found_error": "not_found_error",
            "rate_limit_error": "rate_limit_error",
            "internal_server_error": "internal_server_error",
            "server_error": "overloaded_error",
        }

        anthropic_error_type = error_type_mapping.get(
            error_type, "internal_server_error"
        )

        # Return Anthropic-formatted error
        return {
            "error": {
                "type": anthropic_error_type,
                "message": error_message,
            }
        }

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert OpenAI streaming data to Anthropic streaming format.

        Args:
            stream: OpenAI streaming response iterator

        Returns:
            Anthropic streaming event data
        """
        return self._adapt_stream_impl(stream)

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Implementation of OpenAI to Anthropic streaming conversion."""
        logger = get_logger(__name__)

        try:
            logger.debug(
                "openai_to_anthropic_stream_conversion_started",
                category="streaming_conversion",
            )

            message_started = False
            content_block_started = False
            content_block_index = 0

            async for chunk in stream:
                if not isinstance(chunk, dict):
                    continue

                # Handle OpenAI streaming chunk
                choices = chunk.get("choices", [])
                if not choices:
                    continue

                choice = choices[0]
                delta = choice.get("delta", {})
                finish_reason = choice.get("finish_reason")

                # Send message_start if not sent yet
                if not message_started and (delta or finish_reason):
                    yield {
                        "type": "message_start",
                        "message": {
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": chunk.get("model", "gpt-4"),
                        },
                    }
                    message_started = True

                # Handle role delta (usually first chunk)
                if delta.get("role"):
                    continue  # Already handled in message_start

                # Handle content delta
                if delta.get("content"):
                    if not content_block_started:
                        yield {
                            "type": "content_block_start",
                            "index": content_block_index,
                            "content_block": {"type": "text", "text": ""},
                        }
                        content_block_started = True

                    yield {
                        "type": "content_block_delta",
                        "index": content_block_index,
                        "delta": {"type": "text_delta", "text": delta["content"]},
                    }

                # Handle tool calls delta
                if delta.get("tool_calls"):
                    for tool_call in delta["tool_calls"]:
                        function = tool_call.get("function", {})

                        # Start tool use block
                        if function.get("name") or tool_call.get("id"):
                            if content_block_started:
                                yield {
                                    "type": "content_block_stop",
                                    "index": content_block_index,
                                }
                                content_block_index += 1
                                content_block_started = False

                            yield {
                                "type": "content_block_start",
                                "index": content_block_index,
                                "content_block": {
                                    "type": "tool_use",
                                    "id": tool_call.get("id", ""),
                                    "name": function.get("name", ""),
                                },
                            }
                            content_block_started = True

                        # Stream tool arguments
                        if function.get("arguments"):
                            yield {
                                "type": "content_block_delta",
                                "index": content_block_index,
                                "delta": {
                                    "type": "input_json_delta",
                                    "partial_json": function["arguments"],
                                },
                            }

                # Handle finish reason
                if finish_reason:
                    if content_block_started:
                        yield {
                            "type": "content_block_stop",
                            "index": content_block_index,
                        }

                    # Convert finish reason and send message delta
                    anthropic_stop_reason = self._convert_openai_finish_reason(
                        finish_reason
                    )

                    # Include usage if available
                    message_delta: dict[str, Any] = {
                        "stop_reason": anthropic_stop_reason
                    }

                    usage = chunk.get("usage")
                    if usage:
                        message_delta["usage"] = self._convert_openai_usage(usage)

                    yield {"type": "message_delta", "delta": message_delta}

                    yield {"type": "message_stop"}
                    break

            logger.debug(
                "openai_to_anthropic_stream_conversion_completed",
                category="streaming_conversion",
            )

        except Exception as e:
            logger.error(
                "openai_to_anthropic_stream_conversion_failed",
                error=str(e),
                operation="adapt_stream",
                exc_info=e,
            )
            # Send error event in Anthropic format
            yield {
                "type": "error",
                "error": {"type": "internal_server_error", "message": str(e)},
            }

    def _convert_anthropic_messages_to_openai(
        self, request: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic messages and system field to OpenAI format.

        Args:
            request: Anthropic format request

        Returns:
            OpenAI format messages list
        """
        messages: list[dict[str, Any]] = []

        # Handle system field - convert to system message
        if "system" in request and request["system"]:
            system_content = request["system"]
            if isinstance(system_content, str):
                messages.append({"role": "system", "content": system_content})
            elif isinstance(system_content, list):
                # Handle system as content blocks - extract text
                text_parts = []
                for block in system_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_parts.append(block.get("text", ""))
                if text_parts:
                    messages.append(
                        {"role": "system", "content": "\n\n".join(text_parts)}
                    )

        # Convert messages
        if "messages" in request and request["messages"]:
            for msg in request["messages"]:
                openai_msg = self._convert_anthropic_message_to_openai(msg)
                if openai_msg:
                    if isinstance(openai_msg, list):
                        messages.extend(openai_msg)
                    else:
                        messages.append(openai_msg)

        return messages

    def _convert_anthropic_message_to_openai(
        self, message: dict[str, Any]
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Convert single Anthropic message to OpenAI format.

        Args:
            message: Anthropic message

        Returns:
            OpenAI message(s) or None if invalid
        """
        role = message.get("role")
        content = message.get("content")

        if role in ["user", "assistant"]:
            openai_msg: dict[str, Any] = {"role": role}

            if isinstance(content, str):
                openai_msg["content"] = content
            elif isinstance(content, list):
                # Convert content blocks
                text_content, tool_calls, tool_results = self._convert_content_blocks(
                    content
                )

                if tool_results:
                    # Tool result messages need to be separate tool role messages
                    result_messages = []
                    for tool_result in tool_results:
                        result_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_result.get(
                                    "tool_use_id", "unknown"
                                ),
                                "content": tool_result.get("content", ""),
                            }
                        )

                    # If there's text content, create user message
                    if text_content:
                        result_messages.append(
                            {"role": "user", "content": text_content}
                        )

                    return result_messages

                # Regular message with content and/or tool calls
                if text_content:
                    openai_msg["content"] = text_content
                if tool_calls:
                    openai_msg["tool_calls"] = tool_calls

                # OpenAI requires content to be null when there are tool calls but no text
                if tool_calls and not text_content:
                    openai_msg["content"] = None

            return openai_msg

        return None

    def _convert_content_blocks(
        self, content: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]]]:
        """Convert Anthropic content blocks to OpenAI format.

        Args:
            content: Anthropic content blocks

        Returns:
            Tuple of (text_content, tool_calls, tool_results)
        """
        text_parts = []
        tool_calls = []
        tool_results = []

        for block in content:
            block_type = block.get("type")

            if block_type == "text":
                text_parts.append(block.get("text", ""))
            elif block_type == "thinking":
                # Convert thinking blocks to text with markup
                thinking_text = block.get("thinking", "")
                signature = block.get("signature")
                if thinking_text:
                    thinking_content = (
                        f'<thinking signature="{signature}">{thinking_text}</thinking>'
                    )
                    text_parts.append(thinking_content)
            elif block_type == "image":
                # Convert image to OpenAI format
                source = block.get("source", {})
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    image_url = f"data:{media_type};base64,{data}"
                    # For now, just describe the image in text
                    text_parts.append(f"[Image: {image_url[:100]}...]")
            elif block_type == "tool_use":
                # Convert to OpenAI tool call
                tool_call = {
                    "id": block.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": block.get("name", ""),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
                tool_calls.append(tool_call)
            elif block_type == "tool_result":
                # Collect tool results for separate processing
                tool_results.append(block)

        text_content = "\n\n".join(text_parts) if text_parts else None
        return text_content, tool_calls, tool_results

    def _handle_optional_parameters(
        self, request: dict[str, Any], openai_request: dict[str, Any]
    ) -> None:
        """Handle optional parameters conversion."""
        # Direct mappings
        for field in ["max_tokens", "temperature", "top_p", "stream"]:
            if field in request:
                openai_request[field] = request[field]

        # stop_sequences -> stop
        if "stop_sequences" in request:
            openai_request["stop"] = request["stop_sequences"]

    def _handle_tools(
        self, request: dict[str, Any], openai_request: dict[str, Any]
    ) -> None:
        """Handle tools and tool_choice conversion."""
        # Convert tools
        if "tools" in request:
            openai_request["tools"] = self._convert_anthropic_tools_to_openai(
                request["tools"]
            )

        # Convert tool_choice
        if "tool_choice" in request:
            openai_request["tool_choice"] = (
                self._convert_anthropic_tool_choice_to_openai(request["tool_choice"])
            )

    def _convert_anthropic_tools_to_openai(
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert Anthropic tools to OpenAI format."""
        openai_tools = []

        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {}),
                },
            }
            openai_tools.append(openai_tool)

        return openai_tools

    def _convert_anthropic_tool_choice_to_openai(
        self, tool_choice: dict[str, Any]
    ) -> str | dict[str, Any]:
        """Convert Anthropic tool_choice to OpenAI format."""
        if not isinstance(tool_choice, dict):
            return "auto"

        choice_type = tool_choice.get("type")

        if choice_type == "auto":
            return "auto"
        elif choice_type == "any":
            return "required"  # OpenAI uses "required" for "must use a tool"
        elif choice_type == "tool":
            # Specific tool choice
            tool_name = tool_choice.get("name", "")
            return {"type": "function", "function": {"name": tool_name}}
        elif choice_type == "none":
            return "none"

        return "auto"  # Default fallback

    def _handle_metadata(
        self, request: dict[str, Any], openai_request: dict[str, Any]
    ) -> None:
        """Handle metadata conversion."""
        metadata = request.get("metadata", {})
        if isinstance(metadata, dict) and "user_id" in metadata:
            openai_request["user"] = metadata["user_id"]

    def _convert_openai_choices_to_content(
        self, response: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], str]:
        """Convert OpenAI choices to Anthropic content blocks.

        Args:
            response: OpenAI format response

        Returns:
            Tuple of (content_blocks, stop_reason)
        """
        content_blocks: list[dict[str, Any]] = []
        stop_reason = "end_turn"  # Default

        choices = response.get("choices", [])
        if choices:
            choice = choices[0]  # Take first choice
            message = choice.get("message", {})

            # Convert content
            content = message.get("content")
            if content:
                content_blocks.append({"type": "text", "text": content})

            # Convert tool calls
            tool_calls = message.get("tool_calls", [])
            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                arguments_str = function.get("arguments", "{}")

                # Parse arguments JSON string to dict
                try:
                    if isinstance(arguments_str, str):
                        arguments = json.loads(arguments_str)
                    else:
                        arguments = arguments_str
                except json.JSONDecodeError:
                    arguments = {}

                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.get("id", ""),
                        "name": function.get("name", ""),
                        "input": arguments,
                    }
                )

            # Convert finish reason
            finish_reason = choice.get("finish_reason")
            stop_reason = self._convert_openai_finish_reason(finish_reason)

        # Ensure we always have at least one content block
        if not content_blocks:
            content_blocks.append({"type": "text", "text": ""})

        return content_blocks, stop_reason

    def _convert_openai_finish_reason(self, finish_reason: str | None) -> str:
        """Convert OpenAI finish_reason to Anthropic stop_reason.

        Args:
            finish_reason: OpenAI finish reason

        Returns:
            Anthropic stop reason
        """
        mapping = {
            "stop": "end_turn",
            "length": "max_tokens",
            "tool_calls": "tool_use",
            "content_filter": "stop_sequence",
        }
        return mapping.get(finish_reason or "stop", "end_turn")

    def _convert_openai_usage(self, usage: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI usage to Anthropic format.

        Args:
            usage: OpenAI usage information

        Returns:
            Anthropic format usage
        """
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        }
