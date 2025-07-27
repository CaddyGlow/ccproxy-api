"""OpenAI API adapter implementation.

This module provides the OpenAI adapter that implements the APIAdapter interface
for converting between OpenAI and Anthropic API formats.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any, Literal, cast

import structlog
from pydantic import ValidationError

from ccproxy.core.interfaces import APIAdapter
from ccproxy.utils.model_mapping import map_model_to_claude

from .models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIResponseMessage,
    OpenAIUsage,
    format_openai_tool_call,
    generate_openai_response_id,
    generate_openai_system_fingerprint,
)
from .streaming import OpenAIStreamProcessor


logger = structlog.get_logger(__name__)


class OpenAIAdapter(APIAdapter):
    """OpenAI API adapter for converting between OpenAI and Anthropic formats."""

    def __init__(self, include_sdk_content_as_xml: bool = False) -> None:
        """Initialize the OpenAI adapter."""
        self.include_sdk_content_as_xml = include_sdk_content_as_xml

    def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Convert OpenAI request format to Anthropic format.

        Args:
            request: OpenAI format request

        Returns:
            Anthropic format request

        Raises:
            ValueError: If the request format is invalid or unsupported
        """
        try:
            # Parse OpenAI request
            openai_req = OpenAIChatCompletionRequest(**request)
        except ValidationError as e:
            raise ValueError(f"Invalid OpenAI request format: {e}") from e

        # Map OpenAI model to Claude model
        model = map_model_to_claude(openai_req.model)

        # Convert messages
        messages, system_prompt = self._convert_messages_to_anthropic(
            openai_req.messages
        )

        # Build base Anthropic request
        anthropic_request = {
            "model": model,
            "messages": messages,
            "max_tokens": openai_req.max_tokens or 4096,
        }

        # Add system prompt if present
        if system_prompt:
            anthropic_request["system"] = system_prompt

        # Add optional parameters
        self._handle_optional_parameters(openai_req, anthropic_request)

        # Handle metadata
        self._handle_metadata(openai_req, anthropic_request)

        # Handle response format
        anthropic_request = self._handle_response_format(openai_req, anthropic_request)

        # Handle thinking configuration
        anthropic_request = self._handle_thinking_parameters(
            openai_req, anthropic_request
        )

        # Log unsupported parameters
        self._log_unsupported_parameters(openai_req)

        # Handle tools and tool choice
        self._handle_tools(openai_req, anthropic_request)

        logger.debug(
            "format_conversion_completed",
            from_format="openai",
            to_format="anthropic",
            original_model=openai_req.model,
            anthropic_model=anthropic_request.get("model"),
            has_tools=bool(anthropic_request.get("tools")),
            has_system=bool(anthropic_request.get("system")),
            message_count=len(cast(list[Any], anthropic_request["messages"])),
            operation="adapt_request",
        )
        return anthropic_request

    def _handle_optional_parameters(
        self,
        openai_req: OpenAIChatCompletionRequest,
        anthropic_request: dict[str, Any],
    ) -> None:
        """Handle optional parameters like temperature, top_p, stream, and stop."""
        if openai_req.temperature is not None:
            anthropic_request["temperature"] = openai_req.temperature

        if openai_req.top_p is not None:
            anthropic_request["top_p"] = openai_req.top_p

        if openai_req.stream is not None:
            anthropic_request["stream"] = openai_req.stream

        if openai_req.stop is not None:
            if isinstance(openai_req.stop, str):
                anthropic_request["stop_sequences"] = [openai_req.stop]
            else:
                anthropic_request["stop_sequences"] = openai_req.stop

    def _handle_metadata(
        self,
        openai_req: OpenAIChatCompletionRequest,
        anthropic_request: dict[str, Any],
    ) -> None:
        """Handle metadata and user field combination."""
        metadata = {}
        if openai_req.user:
            metadata["user_id"] = openai_req.user
        if openai_req.metadata:
            metadata.update(openai_req.metadata)
        if metadata:
            anthropic_request["metadata"] = metadata

    def _handle_response_format(
        self,
        openai_req: OpenAIChatCompletionRequest,
        anthropic_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle response format by modifying system prompt for JSON mode."""
        if openai_req.response_format:
            format_type = (
                openai_req.response_format.type if openai_req.response_format else None
            )
            system_prompt = anthropic_request.get("system")

            if format_type == "json_object" and system_prompt is not None:
                system_prompt += "\nYou must respond with valid JSON only."
                anthropic_request["system"] = system_prompt
            elif format_type == "json_schema" and system_prompt is not None:
                # For JSON schema, we can add more specific instructions
                if openai_req.response_format and hasattr(
                    openai_req.response_format, "json_schema"
                ):
                    system_prompt += f"\nYou must respond with valid JSON that conforms to this schema: {openai_req.response_format.json_schema}"
                anthropic_request["system"] = system_prompt

        return anthropic_request

    def _handle_thinking_parameters(
        self,
        openai_req: OpenAIChatCompletionRequest,
        anthropic_request: dict[str, Any],
    ) -> dict[str, Any]:
        """Handle reasoning_effort and thinking configuration for o1/o3 models."""
        # Automatically enable thinking for o1 models even without explicit reasoning_effort
        if (
            openai_req.reasoning_effort
            or openai_req.model.startswith("o1")
            or openai_req.model.startswith("o3")
        ):
            # Map reasoning effort to thinking tokens
            thinking_tokens_map = {
                "low": 1000,
                "medium": 5000,
                "high": 10000,
            }

            # Default thinking tokens based on model if reasoning_effort not specified
            default_thinking_tokens = 5000  # medium by default
            if openai_req.model.startswith("o3"):
                default_thinking_tokens = 10000  # high for o3 models
            elif openai_req.model == "o1-mini":
                default_thinking_tokens = 3000  # lower for mini model

            thinking_tokens = (
                thinking_tokens_map.get(
                    openai_req.reasoning_effort, default_thinking_tokens
                )
                if openai_req.reasoning_effort
                else default_thinking_tokens
            )

            anthropic_request["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_tokens,
            }

            # Ensure max_tokens is greater than budget_tokens
            current_max_tokens = cast(int, anthropic_request.get("max_tokens", 4096))
            if current_max_tokens <= thinking_tokens:
                # Set max_tokens to be 2x thinking tokens + some buffer for response
                anthropic_request["max_tokens"] = thinking_tokens + max(
                    thinking_tokens, 4096
                )
                logger.debug(
                    "max_tokens_adjusted_for_thinking",
                    original_max_tokens=current_max_tokens,
                    thinking_tokens=thinking_tokens,
                    new_max_tokens=anthropic_request["max_tokens"],
                    operation="adapt_request",
                )

            # When thinking is enabled, temperature must be 1.0
            if (
                anthropic_request.get("temperature") is not None
                and anthropic_request["temperature"] != 1.0
            ):
                logger.debug(
                    "temperature_adjusted_for_thinking",
                    original_temperature=anthropic_request["temperature"],
                    new_temperature=1.0,
                    operation="adapt_request",
                )
                anthropic_request["temperature"] = 1.0
            elif "temperature" not in anthropic_request:
                # Set default temperature to 1.0 for thinking mode
                anthropic_request["temperature"] = 1.0

            logger.debug(
                "thinking_enabled",
                reasoning_effort=openai_req.reasoning_effort,
                model=openai_req.model,
                thinking_tokens=thinking_tokens,
                temperature=anthropic_request["temperature"],
                operation="adapt_request",
            )

        return anthropic_request

    def _log_unsupported_parameters(
        self, openai_req: OpenAIChatCompletionRequest
    ) -> None:
        """Log warnings for unsupported OpenAI parameters."""
        if openai_req.seed is not None:
            logger.debug(
                "unsupported_parameter_ignored",
                parameter="seed",
                value=openai_req.seed,
                operation="adapt_request",
            )
        if openai_req.logprobs or openai_req.top_logprobs:
            logger.debug(
                "unsupported_parameters_ignored",
                parameters=["logprobs", "top_logprobs"],
                logprobs=openai_req.logprobs,
                top_logprobs=openai_req.top_logprobs,
                operation="adapt_request",
            )
        if openai_req.store:
            logger.debug(
                "unsupported_parameter_ignored",
                parameter="store",
                value=openai_req.store,
                operation="adapt_request",
            )

    def _handle_tools(
        self,
        openai_req: OpenAIChatCompletionRequest,
        anthropic_request: dict[str, Any],
    ) -> None:
        """Handle tools, functions, and tool choice conversion."""
        # Handle tools/functions
        if openai_req.tools:
            anthropic_request["tools"] = self._convert_tools_to_anthropic(
                openai_req.tools
            )
        elif openai_req.functions:
            # Convert deprecated functions to tools
            anthropic_request["tools"] = self._convert_functions_to_anthropic(
                openai_req.functions
            )

        # Handle tool choice
        if openai_req.tool_choice:
            # Convert tool choice - can be string or OpenAIToolChoice object
            if isinstance(openai_req.tool_choice, str):
                anthropic_request["tool_choice"] = (
                    self._convert_tool_choice_to_anthropic(openai_req.tool_choice)
                )
            else:
                # Convert OpenAIToolChoice object to dict
                tool_choice_dict = {
                    "type": openai_req.tool_choice.type,
                    "function": openai_req.tool_choice.function,
                }
                anthropic_request["tool_choice"] = (
                    self._convert_tool_choice_to_anthropic(tool_choice_dict)
                )
        elif openai_req.function_call:
            # Convert deprecated function_call to tool_choice
            anthropic_request["tool_choice"] = self._convert_function_call_to_anthropic(
                openai_req.function_call
            )

    def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic response format to OpenAI format.

        Args:
            response: Anthropic format response

        Returns:
            OpenAI format response

        Raises:
            ValueError: If the response format is invalid or unsupported
        """
        try:
            # Extract original model from response metadata if available
            original_model = response.get("model", "gpt-4")

            # Generate response ID
            request_id = generate_openai_response_id()

            # Convert content and extract tool calls
            content, tool_calls = self._convert_content_blocks(response)

            # Create OpenAI message
            message = self._create_openai_message(content, tool_calls)

            # Create choice with proper finish reason
            choice = self._create_openai_choice(message, response)

            # Create usage information
            usage = self._create_openai_usage(response)

            # Create final OpenAI response
            openai_response = OpenAIChatCompletionResponse(
                id=request_id,
                object="chat.completion",
                created=int(time.time()),
                model=original_model,
                choices=[choice],
                usage=usage,
                system_fingerprint=generate_openai_system_fingerprint(),
            )

            logger.debug(
                "format_conversion_completed",
                from_format="anthropic",
                to_format="openai",
                response_id=request_id,
                original_model=original_model,
                finish_reason=choice.finish_reason,
                content_length=len(content) if content else 0,
                tool_calls_count=len(tool_calls),
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                operation="adapt_response",
                choice=choice,
            )
            return openai_response.model_dump()

        except ValidationError as e:
            raise ValueError(f"Invalid Anthropic response format: {e}") from e

    def _convert_content_blocks(
        self, response: dict[str, Any]
    ) -> tuple[str, list[Any]]:
        """Convert Anthropic content blocks to OpenAI format content and tool calls."""
        content = ""
        tool_calls: list[Any] = []

        if "content" in response and response["content"]:
            for block in response["content"]:
                if block.get("type") == "text":
                    text_content = block.get("text", "")
                    # Forward text content as-is (already formatted if needed)
                    content += text_content
                elif block.get("type") == "system_message":
                    # Handle custom system_message content blocks
                    system_text = block.get("text", "")
                    source = block.get("source", "claude_code_sdk")
                    # Format as text with clear source attribution
                    content += f"[{source}]: {system_text}"
                elif block.get("type") == "tool_use_sdk":
                    # Handle custom tool_use_sdk content blocks - convert to standard tool_calls
                    tool_call_block = {
                        "type": "tool_use",
                        "id": block.get("id", ""),
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                    }
                    tool_calls.append(format_openai_tool_call(tool_call_block))
                elif block.get("type") == "tool_result_sdk":
                    # Handle custom tool_result_sdk content blocks - add as text with source attribution
                    source = block.get("source", "claude_code_sdk")
                    tool_use_id = block.get("tool_use_id", "")
                    result_content = block.get("content", "")
                    is_error = block.get("is_error", False)
                    error_indicator = " (ERROR)" if is_error else ""
                    content += f"[{source} tool_result {tool_use_id}{error_indicator}]: {result_content}"
                elif block.get("type") == "result_message":
                    # Handle custom result_message content blocks - add as text with source attribution
                    source = block.get("source", "claude_code_sdk")
                    result_data = block.get("data", {})
                    session_id = result_data.get("session_id", "")
                    stop_reason = result_data.get("stop_reason", "")
                    usage = result_data.get("usage", {})
                    cost_usd = result_data.get("total_cost_usd")
                    formatted_text = f"[{source} result {session_id}]: stop_reason={stop_reason}, usage={usage}"
                    if cost_usd is not None:
                        formatted_text += f", cost_usd={cost_usd}"
                    content += formatted_text
                elif block.get("type") == "thinking":
                    # Handle thinking blocks - we can include them with a marker
                    thinking_text = block.get("thinking", "")
                    signature = block.get("signature")
                    if thinking_text:
                        content += f'<thinking signature="{signature}">{thinking_text}</thinking>'
                elif block.get("type") == "tool_use":
                    # Handle legacy tool_use content blocks
                    tool_calls.append(format_openai_tool_call(block))
                else:
                    logger.warning(
                        "unsupported_content_block_type", type=block.get("type")
                    )

        return content, tool_calls

    def _create_openai_message(
        self, content: str, tool_calls: list[Any]
    ) -> OpenAIResponseMessage:
        """Create OpenAI message with proper content handling."""
        # When there are tool calls but no content, use empty string instead of None
        # Otherwise, if content is empty string, convert to None
        final_content: str | None = content
        if tool_calls and not content:
            final_content = ""
        elif content == "":
            final_content = None

        return OpenAIResponseMessage(
            role="assistant",
            content=final_content,
            tool_calls=tool_calls if tool_calls else None,
        )

    def _create_openai_choice(
        self, message: OpenAIResponseMessage, response: dict[str, Any]
    ) -> OpenAIChoice:
        """Create OpenAI choice with proper finish reason handling."""
        # Map stop reason
        finish_reason = self._convert_stop_reason_to_openai(response.get("stop_reason"))

        # Ensure finish_reason is a valid literal type
        if finish_reason not in ["stop", "length", "tool_calls", "content_filter"]:
            finish_reason = "stop"

        # Cast to proper literal type
        valid_finish_reason = cast(
            Literal["stop", "length", "tool_calls", "content_filter"], finish_reason
        )

        return OpenAIChoice(
            index=0,
            message=message,
            finish_reason=valid_finish_reason,
            logprobs=None,  # Anthropic doesn't support logprobs
        )

    def _create_openai_usage(self, response: dict[str, Any]) -> OpenAIUsage:
        """Create OpenAI usage information from Anthropic response."""
        usage_info = response.get("usage", {})
        return OpenAIUsage(
            prompt_tokens=usage_info.get("input_tokens", 0),
            completion_tokens=usage_info.get("output_tokens", 0),
            total_tokens=usage_info.get("input_tokens", 0)
            + usage_info.get("output_tokens", 0),
        )

    async def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert Anthropic streaming response to OpenAI streaming format.

        Args:
            stream: Anthropic streaming response

        Yields:
            OpenAI format streaming chunks

        Raises:
            ValueError: If the stream format is invalid or unsupported
        """
        # Create stream processor with dict output format
        processor = OpenAIStreamProcessor(
            enable_usage=True,
            enable_tool_calls=True,
            output_format="dict",  # Output dict objects instead of SSE strings
        )

        try:
            # Process the stream - now yields dict objects directly
            async for chunk in processor.process_stream(stream):
                yield chunk  # type: ignore[misc]  # chunk is guaranteed to be dict when output_format="dict"
        except Exception as e:
            logger.error(
                "streaming_conversion_failed",
                error=str(e),
                error_type=type(e).__name__,
                operation="adapt_stream",
                exc_info=True,
            )
            raise ValueError(f"Error processing streaming response: {e}") from e

    def _convert_messages_to_anthropic(
        self, openai_messages: list[Any]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert OpenAI messages to Anthropic format."""
        messages = []
        system_prompt = None

        for msg in openai_messages:
            if msg.role in ["system", "developer"]:
                # System and developer messages become system prompt
                if isinstance(msg.content, str):
                    if system_prompt:
                        system_prompt += "\n" + msg.content
                    else:
                        system_prompt = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts: list[str] = []
                    for block in msg.content:
                        if (
                            hasattr(block, "type")
                            and block.type == "text"
                            and hasattr(block, "text")
                            and block.text
                        ):
                            text_parts.append(block.text)
                    text_content = " ".join(text_parts)
                    if system_prompt:
                        system_prompt += "\n" + text_content
                    else:
                        system_prompt = text_content

            elif msg.role in ["user", "assistant"]:
                # Convert user/assistant messages
                anthropic_msg = {
                    "role": msg.role,
                    "content": self._convert_content_to_anthropic(msg.content),
                }

                # Add tool calls if present
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    # Ensure content is a list
                    if isinstance(anthropic_msg["content"], str):
                        anthropic_msg["content"] = [
                            {"type": "text", "text": anthropic_msg["content"]}
                        ]
                    if not isinstance(anthropic_msg["content"], list):
                        anthropic_msg["content"] = []

                    # Content is now guaranteed to be a list
                    content_list = anthropic_msg["content"]
                    for tool_call in msg.tool_calls:
                        content_list.append(
                            self._convert_tool_call_to_anthropic(tool_call)
                        )

                messages.append(anthropic_msg)

            elif msg.role == "tool":
                # Tool result messages
                if messages and messages[-1]["role"] == "user":
                    # Add to previous user message
                    if isinstance(messages[-1]["content"], str):
                        messages[-1]["content"] = [
                            {"type": "text", "text": messages[-1]["content"]}
                        ]

                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": getattr(msg, "tool_call_id", "unknown")
                        or "unknown",
                        "content": msg.content or "",
                    }
                    if isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append(tool_result)
                else:
                    # Create new user message with tool result
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": getattr(msg, "tool_call_id", "unknown")
                        or "unknown",
                        "content": msg.content or "",
                    }
                    messages.append(
                        {
                            "role": "user",
                            "content": [tool_result],
                        }
                    )

        return messages, system_prompt

    def _convert_content_to_anthropic(
        self, content: str | list[Any] | None
    ) -> str | list[dict[str, Any]]:
        """Convert OpenAI content to Anthropic format."""
        if content is None:
            return ""

        if isinstance(content, str):
            # Check if the string contains thinking blocks
            thinking_pattern = r'<thinking signature="([^"]*)">(.*?)</thinking>'
            matches = re.findall(thinking_pattern, content, re.DOTALL)

            if matches:
                # Convert string with thinking blocks to list format
                anthropic_content: list[dict[str, Any]] = []
                last_end = 0

                for match in re.finditer(thinking_pattern, content, re.DOTALL):
                    # Add any text before the thinking block
                    if match.start() > last_end:
                        text_before = content[last_end : match.start()].strip()
                        if text_before:
                            anthropic_content.append(
                                {"type": "text", "text": text_before}
                            )

                    # Add the thinking block
                    signature = match.group(1)
                    thinking_text = match.group(2)
                    thinking_block: dict[str, Any] = {
                        "type": "thinking",
                        "thinking": thinking_text,  # Changed from "text" to "thinking"
                    }
                    if signature and signature != "None":
                        thinking_block["signature"] = signature
                    anthropic_content.append(thinking_block)

                    last_end = match.end()

                # Add any remaining text after the last thinking block
                if last_end < len(content):
                    remaining_text = content[last_end:].strip()
                    if remaining_text:
                        anthropic_content.append(
                            {"type": "text", "text": remaining_text}
                        )

                return anthropic_content
            else:
                return content

        # content must be a list at this point
        anthropic_content = []
        for block in content:
            # Handle both Pydantic objects and dicts
            if hasattr(block, "type"):
                # This is a Pydantic object
                block_type = getattr(block, "type", None)
                if (
                    block_type == "text"
                    and hasattr(block, "text")
                    and block.text is not None
                ):
                    anthropic_content.append(
                        {
                            "type": "text",
                            "text": block.text,
                        }
                    )
                elif (
                    block_type == "image_url"
                    and hasattr(block, "image_url")
                    and block.image_url is not None
                ):
                    # Get URL from image_url
                    if hasattr(block.image_url, "url"):
                        url = block.image_url.url
                    elif isinstance(block.image_url, dict):
                        url = block.image_url.get("url", "")
                    else:
                        url = ""

                    if url.startswith("data:"):
                        # Base64 encoded image
                        try:
                            media_type, data = url.split(";base64,")
                            media_type = media_type.split(":")[1]
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
                        except ValueError:
                            logger.warning(
                                "invalid_base64_image_url",
                                url=url[:100] + "..." if len(url) > 100 else url,
                                operation="convert_content_to_anthropic",
                            )
                    else:
                        # URL-based image (not directly supported by Anthropic)
                        anthropic_content.append(
                            {
                                "type": "text",
                                "text": f"[Image: {url}]",
                            }
                        )
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    anthropic_content.append(
                        {
                            "type": "text",
                            "text": block.get("text", ""),
                        }
                    )
                elif block.get("type") == "image_url":
                    # Convert image URL to Anthropic format
                    image_url = block.get("image_url", {})
                    url = image_url.get("url", "")

                    if url.startswith("data:"):
                        # Base64 encoded image
                        try:
                            media_type, data = url.split(";base64,")
                            media_type = media_type.split(":")[1]
                            anthropic_content.append(
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": data,
                                    },
                                }
                            )
                        except ValueError:
                            logger.warning(
                                "invalid_base64_image_url",
                                url=url[:100] + "..." if len(url) > 100 else url,
                                operation="convert_content_to_anthropic",
                            )
                    else:
                        # URL-based image (not directly supported by Anthropic)
                        anthropic_content.append(
                            {
                                "type": "text",
                                "text": f"[Image: {url}]",
                            }
                        )

        return anthropic_content if anthropic_content else ""

    def _convert_tools_to_anthropic(
        self, tools: list[dict[str, Any]] | list[Any]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            # Handle both dict and Pydantic model cases
            if isinstance(tool, dict):
                if tool.get("type") == "function":
                    func = tool.get("function", {})
                    anthropic_tools.append(
                        {
                            "name": func.get("name", ""),
                            "description": func.get("description", ""),
                            "input_schema": func.get("parameters", {}),
                        }
                    )
            elif hasattr(tool, "type") and tool.type == "function":
                # Handle Pydantic OpenAITool model
                anthropic_tools.append(
                    {
                        "name": tool.function.name,
                        "description": tool.function.description or "",
                        "input_schema": tool.function.parameters,
                    }
                )

        return anthropic_tools

    def _convert_functions_to_anthropic(
        self, functions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI functions to Anthropic tools format."""
        anthropic_tools = []

        for func in functions:
            anthropic_tools.append(
                {
                    "name": func.get("name", ""),
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {}),
                }
            )

        return anthropic_tools

    def _convert_tool_choice_to_anthropic(
        self, tool_choice: str | dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI tool_choice to Anthropic format."""
        if isinstance(tool_choice, str):
            mapping = {
                "none": {"type": "none"},
                "auto": {"type": "auto"},
                "required": {"type": "any"},
            }
            return mapping.get(tool_choice, {"type": "auto"})

        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            func = tool_choice.get("function", {})
            return {
                "type": "tool",
                "name": func.get("name", ""),
            }

        return {"type": "auto"}

    def _convert_function_call_to_anthropic(
        self, function_call: str | dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI function_call to Anthropic tool_choice format."""
        if isinstance(function_call, str):
            if function_call == "none":
                return {"type": "none"}
            elif function_call == "auto":
                return {"type": "auto"}

        elif isinstance(function_call, dict):
            return {
                "type": "tool",
                "name": function_call.get("name", ""),
            }

        return {"type": "auto"}

    def _convert_tool_call_to_anthropic(
        self, tool_call: dict[str, Any]
    ) -> dict[str, Any]:
        """Convert OpenAI tool call to Anthropic format."""
        func = tool_call.get("function", {})

        # Parse arguments string to dict for Anthropic format
        arguments_str = func.get("arguments", "{}")
        try:
            if isinstance(arguments_str, str):
                input_dict = json.loads(arguments_str)
            else:
                input_dict = arguments_str  # Already a dict
        except json.JSONDecodeError:
            logger.warning(
                "tool_arguments_parse_failed",
                arguments=arguments_str[:200] + "..."
                if len(str(arguments_str)) > 200
                else str(arguments_str),
                operation="convert_tool_call_to_anthropic",
            )
            input_dict = {}

        return {
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": func.get("name", ""),
            "input": input_dict,
        }

    def _convert_stop_reason_to_openai(self, stop_reason: str | None) -> str | None:
        """Convert Anthropic stop reason to OpenAI format."""
        if stop_reason is None:
            return None

        mapping = {
            "end_turn": "stop",
            "max_tokens": "length",
            "stop_sequence": "stop",
            "tool_use": "tool_calls",
            "pause_turn": "stop",
            "refusal": "content_filter",
        }

        return mapping.get(stop_reason, "stop")

    def adapt_error(self, error_body: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic error format to OpenAI error format.

        Args:
            error_body: Anthropic error response

        Returns:
            OpenAI-formatted error response
        """
        # Extract error details from Anthropic format
        anthropic_error = error_body.get("error", {})
        error_type = anthropic_error.get("type", "internal_server_error")
        error_message = anthropic_error.get("message", "An error occurred")

        # Map Anthropic error types to OpenAI error types
        error_type_mapping = {
            "invalid_request_error": "invalid_request_error",
            "authentication_error": "invalid_request_error",
            "permission_error": "invalid_request_error",
            "not_found_error": "invalid_request_error",
            "rate_limit_error": "rate_limit_error",
            "internal_server_error": "internal_server_error",
            "overloaded_error": "server_error",
        }

        openai_error_type = error_type_mapping.get(error_type, "invalid_request_error")

        # Return OpenAI-formatted error
        return {
            "error": {
                "message": error_message,
                "type": openai_error_type,
                "code": error_type,  # Preserve original error type as code
            }
        }


__all__ = [
    "OpenAIAdapter",
    "OpenAIChatCompletionRequest",
    "OpenAIChatCompletionResponse",
]
