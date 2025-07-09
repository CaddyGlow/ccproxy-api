"""Translation layer for converting between OpenAI and Anthropic formats."""

import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from ccproxy.models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIMessage,
    OpenAIMessageContent,
    OpenAIResponseMessage,
    OpenAIStreamingChatCompletionResponse,
    OpenAIStreamingChoice,
    OpenAIStreamingDelta,
    OpenAITool,
    OpenAIToolCall,
    OpenAIUsage,
)
from ccproxy.utils.logging import get_logger


logger = get_logger(__name__)


# OpenAI to Claude model mapping (startswith matching)
OPENAI_TO_CLAUDE_MODEL_MAPPING = {
    "gpt-4o-mini": "claude-3-5-haiku-latest",
    "o3-mini": "claude-opus-4-20250514",
    "o1-mini": "claude-sonnet-4-20250514",
    "gpt-4o": "claude-3-7-sonnet-20250219",
}


def map_openai_model_to_claude(model: str) -> str:
    """
    Map OpenAI model names to Claude models using startswith matching.

    Args:
        model: OpenAI model name

    Returns:
        Mapped Claude model name or original if no mapping found
    """
    # Pass through Claude models without mapping
    if model.startswith("claude-"):
        return model

    # Check for exact matches first
    if model in OPENAI_TO_CLAUDE_MODEL_MAPPING:
        return OPENAI_TO_CLAUDE_MODEL_MAPPING[model]

    # Check for startswith matches
    for openai_prefix, claude_model in OPENAI_TO_CLAUDE_MODEL_MAPPING.items():
        if model.startswith(openai_prefix):
            return claude_model

    # Return original model if no mapping found
    return model


class OpenAITranslator:
    """Translator for converting between OpenAI and Anthropic formats."""

    def __init__(self) -> None:
        """Initialize the translator."""
        pass

    def openai_to_anthropic_request(
        self, openai_request: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Convert OpenAI request format to Anthropic format.

        Args:
            openai_request: OpenAI format request

        Returns:
            Anthropic format request
        """
        # Parse OpenAI request
        openai_req = OpenAIChatCompletionRequest(**openai_request)

        # Map OpenAI model to Claude model
        model = map_openai_model_to_claude(openai_req.model)

        # Convert messages
        messages, system_prompt = self._convert_messages_to_anthropic(
            openai_req.messages
        )

        # Build Anthropic request
        anthropic_request = {
            "model": model,
            "messages": messages,
            "max_tokens": openai_req.max_tokens or 4096,
        }

        # Add system prompt if present
        if system_prompt:
            anthropic_request["system"] = system_prompt

        # Add optional parameters
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

        # Handle metadata - combine user field and metadata
        metadata = {}
        if openai_req.user:
            metadata["user_id"] = openai_req.user
        if openai_req.metadata:
            metadata.update(openai_req.metadata)
        if metadata:
            anthropic_request["metadata"] = metadata

        # Handle response format - add to system prompt for JSON mode
        if openai_req.response_format:
            # response_format is OpenAIResponseFormat object, not a dict
            format_type = (
                openai_req.response_format.type if openai_req.response_format else None
            )

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

        # Handle reasoning_effort (o1 models) -> thinking configuration
        if openai_req.reasoning_effort:
            # Map reasoning effort to thinking tokens
            # These are approximate mappings
            thinking_tokens_map = {
                "low": 1000,
                "medium": 5000,
                "high": 10000,
            }
            thinking_tokens = thinking_tokens_map.get(openai_req.reasoning_effort, 5000)
            anthropic_request["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_tokens,
            }
            logger.debug(
                f"Converted reasoning_effort '{openai_req.reasoning_effort}' to thinking budget {thinking_tokens}"
            )

        # Note: seed, logprobs, top_logprobs, and store don't have direct Anthropic equivalents
        # We'll log if these are requested
        if openai_req.seed is not None:
            logger.debug(
                f"Seed parameter ({openai_req.seed}) requested but not supported by Anthropic"
            )
        if openai_req.logprobs or openai_req.top_logprobs:
            logger.debug("Log probabilities requested but not supported by Anthropic")
        if openai_req.store:
            logger.debug("Store parameter requested but not supported by Anthropic")

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

        logger.debug(f"Converted OpenAI request to Anthropic: {anthropic_request}")
        return anthropic_request

    def anthropic_to_openai_response(
        self,
        anthropic_response: dict[str, Any],
        original_model: str,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Convert Anthropic response format to OpenAI format.

        Args:
            anthropic_response: Anthropic format response
            original_model: Original model requested in OpenAI format
            request_id: Request ID for the response

        Returns:
            OpenAI format response
        """
        import time

        # Generate response ID if not provided
        if request_id is None:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        # Return the original model name as-is
        response_model = original_model

        # Convert content
        content = ""
        tool_calls = []

        if "content" in anthropic_response and anthropic_response["content"]:
            for block in anthropic_response["content"]:
                if block.get("type") == "text":
                    content += block.get("text", "")
                elif block.get("type") == "thinking":
                    # Handle thinking blocks - we can include them with a marker
                    # or skip them entirely. For now, let's include with a marker
                    thinking_text = block.get("text", "")
                    if thinking_text:
                        content += f"[Thinking]\n{thinking_text}\n---\n"
                elif block.get("type") == "tool_use":
                    tool_calls.append(self._convert_tool_use_to_openai(block))

        # Create OpenAI message using the proper model
        message = OpenAIResponseMessage(
            role="assistant",
            content=content or None,
            tool_calls=[OpenAIToolCall(**tc) for tc in tool_calls]
            if tool_calls
            else None,
        )

        # Map stop reason
        finish_reason = self._convert_stop_reason_to_openai(
            anthropic_response.get("stop_reason")
        )

        # Create choice using the proper model
        # Ensure finish_reason is a valid literal type
        if finish_reason not in ["stop", "length", "tool_calls", "content_filter"]:
            finish_reason = "stop"

        # Cast to proper literal type
        valid_finish_reason = cast(
            Literal["stop", "length", "tool_calls", "content_filter"], finish_reason
        )

        choice = OpenAIChoice(
            index=0,
            message=message,
            finish_reason=valid_finish_reason,
            logprobs=None,  # Anthropic doesn't support logprobs
        )

        # Create usage using the proper model
        usage_info = anthropic_response.get("usage", {})
        usage = OpenAIUsage(
            prompt_tokens=usage_info.get("input_tokens", 0),
            completion_tokens=usage_info.get("output_tokens", 0),
            total_tokens=usage_info.get("input_tokens", 0)
            + usage_info.get("output_tokens", 0),
        )

        # Generate system fingerprint
        system_fingerprint = f"fp_{uuid.uuid4().hex[:8]}"

        # Create OpenAI response using the proper model
        openai_response = OpenAIChatCompletionResponse(
            id=request_id,
            object="chat.completion",
            created=int(time.time()),
            model=response_model,
            choices=[choice],
            usage=usage,
            system_fingerprint=system_fingerprint,
        )

        logger.debug(f"Converted Anthropic response to OpenAI: {openai_response}")
        return openai_response.model_dump()

    async def anthropic_to_openai_stream(
        self,
        anthropic_stream: AsyncGenerator[dict[str, Any], None],
        original_model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Convert Anthropic streaming response to OpenAI streaming format.

        This method now uses the unified stream transformer for consistent behavior.

        Args:
            anthropic_stream: Anthropic streaming response
            original_model: Original model requested in OpenAI format
            request_id: Request ID for the response

        Yields:
            OpenAI format streaming chunks
        """
        import json
        import time

        from ccproxy.formatters.stream_transformer import (
            OpenAIStreamTransformer,
            StreamingConfig,
        )

        # Generate response ID if not provided
        if request_id is None:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        # Configure streaming for translator use
        config = StreamingConfig(
            enable_text_chunking=False,  # Keep text as-is for translator
            enable_tool_calls=True,
            enable_usage_info=True,
            chunk_delay_ms=0,  # No artificial delays
            chunk_size_words=1,
        )

        # Create transformer
        transformer = OpenAIStreamTransformer.from_claude_sdk(
            anthropic_stream,
            message_id=request_id,
            model=original_model,
            created=int(time.time()),
            config=config,
        )

        # Transform and yield as dict objects
        async for chunk in transformer.transform():
            # Parse the SSE format string back to dict for compatibility
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        yield json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {data_str}")
                        continue

    def _convert_messages_to_anthropic(
        self, openai_messages: list[OpenAIMessage]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert OpenAI messages to Anthropic format."""
        messages = []
        system_prompt = None

        for msg in openai_messages:
            if msg.role in ["system", "developer"]:
                # System and developer messages become system prompt
                # Developer messages are o1-specific but treated the same as system
                if isinstance(msg.content, str):
                    if system_prompt:
                        system_prompt += "\n" + msg.content
                    else:
                        system_prompt = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts: list[str] = []
                    for block in msg.content:
                        # OpenAIMessageContent objects only
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
                if msg.tool_calls:
                    # Ensure content is a list
                    if isinstance(anthropic_msg["content"], str):
                        anthropic_msg["content"] = [
                            {"type": "text", "text": anthropic_msg["content"]}
                        ]
                    # At this point content is either a list or empty string
                    if not isinstance(anthropic_msg["content"], list):
                        anthropic_msg["content"] = []

                    # Content is now guaranteed to be a list
                    content_list = anthropic_msg["content"]
                    # Type assertion for mypy
                    assert isinstance(content_list, list)
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
                        "tool_use_id": msg.tool_call_id or "unknown",
                        "content": msg.content or "",
                    }
                    # Ensure content is a list before appending
                    if isinstance(messages[-1]["content"], list):
                        messages[-1]["content"].append(tool_result)
                else:
                    # Create new user message with tool result
                    tool_result = {
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "unknown",
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
            return content

        # content must be a list at this point
        anthropic_content = []
        for block in content:
            # Handle both OpenAIMessageContent objects and dicts
            if hasattr(block, "type"):
                # This is an OpenAIMessageContent object
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
                            logger.warning(f"Invalid base64 image URL: {url}")
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
                            logger.warning(f"Invalid base64 image URL: {url}")
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
        self, tools: list[dict[str, Any]] | list[OpenAITool]
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
        import json

        func = tool_call.get("function", {})

        # Parse arguments string to dict for Anthropic format
        arguments_str = func.get("arguments", "{}")
        try:
            if isinstance(arguments_str, str):
                input_dict = json.loads(arguments_str)
            else:
                input_dict = arguments_str  # Already a dict
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool arguments as JSON: {arguments_str}")
            input_dict = {}

        return {
            "type": "tool_use",
            "id": tool_call.get("id", ""),
            "name": func.get("name", ""),
            "input": input_dict,
        }

    def _convert_tool_use_to_openai(self, tool_use: dict[str, Any]) -> dict[str, Any]:
        """Convert Anthropic tool use to OpenAI format."""
        import json

        # Convert input to JSON string if it's a dict, otherwise stringify
        tool_input = tool_use.get("input", {})
        if isinstance(tool_input, dict):
            arguments_str = json.dumps(tool_input)
        else:
            arguments_str = str(tool_input)

        return {
            "id": tool_use.get("id", ""),
            "type": "function",
            "function": {
                "name": tool_use.get("name", ""),
                "arguments": arguments_str,
            },
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
