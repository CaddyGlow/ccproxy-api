"""Translation layer for converting between OpenAI and Anthropic formats."""

import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal

from pydantic import BaseModel, Field

from claude_code_proxy.utils.logging import get_logger


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


class OpenAIMessage(BaseModel):
    """OpenAI message format."""

    role: Literal["system", "user", "assistant", "tool"] = Field(
        ..., description="The role of the message sender"
    )
    content: str | list[dict[str, Any]] | None = Field(
        None, description="The content of the message"
    )
    name: str | None = Field(None, description="Name of the message sender")
    tool_calls: list[dict[str, Any]] | None = Field(
        None, description="Tool calls made by the assistant"
    )
    tool_call_id: str | None = Field(None, description="ID of the tool call")


class OpenAIRequest(BaseModel):
    """OpenAI chat completion request format."""

    model: str = Field(..., description="The model to use")
    messages: list[OpenAIMessage] = Field(
        ..., description="List of messages in the conversation"
    )
    max_tokens: int | None = Field(None, description="Maximum tokens to generate")
    temperature: float | None = Field(None, description="Sampling temperature")
    top_p: float | None = Field(None, description="Nucleus sampling parameter")
    n: int | None = Field(None, description="Number of completions to generate")
    stream: bool | None = Field(False, description="Whether to stream responses")
    stop: str | list[str] | None = Field(None, description="Stop sequences")
    presence_penalty: float | None = Field(None, description="Presence penalty")
    frequency_penalty: float | None = Field(None, description="Frequency penalty")
    logit_bias: dict[str, float] | None = Field(None, description="Logit bias")
    user: str | None = Field(None, description="User identifier")
    functions: list[dict[str, Any]] | None = Field(
        None, description="Available functions (deprecated)"
    )
    function_call: str | dict[str, Any] | None = Field(
        None, description="Function call preference (deprecated)"
    )
    tools: list[dict[str, Any]] | None = Field(None, description="Available tools")
    tool_choice: str | dict[str, Any] | None = Field(
        None, description="Tool choice preference"
    )


class OpenAIChoice(BaseModel):
    """OpenAI choice format."""

    index: int = Field(..., description="Choice index")
    message: OpenAIMessage = Field(..., description="The message")
    finish_reason: str | None = Field(None, description="Reason for finishing")


class OpenAIUsage(BaseModel):
    """OpenAI usage format."""

    prompt_tokens: int = Field(..., description="Number of prompt tokens")
    completion_tokens: int = Field(..., description="Number of completion tokens")
    total_tokens: int = Field(..., description="Total number of tokens")


class OpenAIResponse(BaseModel):
    """OpenAI chat completion response format."""

    id: str = Field(..., description="Response ID")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[OpenAIChoice] = Field(..., description="List of choices")
    usage: OpenAIUsage = Field(..., description="Usage information")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


class OpenAIStreamChoice(BaseModel):
    """OpenAI streaming choice format."""

    index: int = Field(..., description="Choice index")
    delta: dict[str, Any] = Field(..., description="Delta content")
    finish_reason: str | None = Field(None, description="Reason for finishing")


class OpenAIStreamResponse(BaseModel):
    """OpenAI streaming response format."""

    id: str = Field(..., description="Response ID")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: list[OpenAIStreamChoice] = Field(..., description="List of choices")
    usage: OpenAIUsage | None = Field(None, description="Usage information")
    system_fingerprint: str | None = Field(None, description="System fingerprint")


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
        openai_req = OpenAIRequest(**openai_request)

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
            anthropic_request["tool_choice"] = self._convert_tool_choice_to_anthropic(
                openai_req.tool_choice
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
                elif block.get("type") == "tool_use":
                    tool_calls.append(self._convert_tool_use_to_openai(block))

        # Create OpenAI message
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content or None,
        }

        if tool_calls:
            message["tool_calls"] = tool_calls

        # Map stop reason
        finish_reason = self._convert_stop_reason_to_openai(
            anthropic_response.get("stop_reason")
        )

        # Create choice
        choice = {
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
        }

        # Create usage
        usage_info = anthropic_response.get("usage", {})
        usage = {
            "prompt_tokens": usage_info.get("input_tokens", 0),
            "completion_tokens": usage_info.get("output_tokens", 0),
            "total_tokens": usage_info.get("input_tokens", 0)
            + usage_info.get("output_tokens", 0),
        }

        # Create OpenAI response
        openai_response = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": response_model,
            "choices": [choice],
            "usage": usage,
        }

        logger.debug(f"Converted Anthropic response to OpenAI: {openai_response}")
        return openai_response

    async def anthropic_to_openai_stream(
        self,
        anthropic_stream: AsyncIterator[dict[str, Any]],
        original_model: str,
        request_id: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """
        Convert Anthropic streaming response to OpenAI streaming format.

        Args:
            anthropic_stream: Anthropic streaming response
            original_model: Original model requested in OpenAI format
            request_id: Request ID for the response

        Yields:
            OpenAI format streaming chunks
        """
        import time

        # Generate response ID if not provided
        if request_id is None:
            request_id = f"chatcmpl-{uuid.uuid4().hex[:29]}"

        # Return the original model name as-is
        response_model = original_model

        created = int(time.time())
        tool_calls = []
        current_tool_call_index = 0

        async for chunk in anthropic_stream:
            chunk_type = chunk.get("type")

            if chunk_type == "message_start":
                # Send initial chunk
                yield {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": response_model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None,
                        }
                    ],
                }

            elif chunk_type == "content_block_start":
                block = chunk.get("content_block", {})
                if block.get("type") == "tool_use":
                    # Start of tool use
                    tool_call = {
                        "index": current_tool_call_index,
                        "id": block.get("id", f"call_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": "",
                        },
                    }
                    tool_calls.append(tool_call)

                    yield {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"tool_calls": [tool_call]},
                                "finish_reason": None,
                            }
                        ],
                    }

                    current_tool_call_index += 1

            elif chunk_type == "content_block_delta":
                delta = chunk.get("delta", {})
                if delta.get("type") == "text_delta":
                    # Text content delta
                    text = delta.get("text", "")
                    yield {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": None,
                            }
                        ],
                    }
                elif delta.get("type") == "input_json_delta":
                    # Tool input delta
                    if tool_calls:
                        partial_json = delta.get("partial_json", "")
                        yield {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": response_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": len(tool_calls) - 1,
                                                "function": {"arguments": partial_json},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }

            elif chunk_type == "message_delta":
                delta = chunk.get("delta", {})
                finish_reason = self._convert_stop_reason_to_openai(
                    delta.get("stop_reason")
                )

                if finish_reason:
                    # Final chunk
                    usage_info = chunk.get("usage", {})
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": response_model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_reason,
                            }
                        ],
                    }

                    if usage_info:
                        final_chunk["usage"] = {
                            "prompt_tokens": usage_info.get("input_tokens", 0),
                            "completion_tokens": usage_info.get("output_tokens", 0),
                            "total_tokens": usage_info.get("input_tokens", 0)
                            + usage_info.get("output_tokens", 0),
                        }

                    yield final_chunk

    def _convert_messages_to_anthropic(
        self, openai_messages: list[OpenAIMessage]
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Convert OpenAI messages to Anthropic format."""
        messages = []
        system_prompt = None

        for msg in openai_messages:
            if msg.role == "system":
                # System messages become system prompt
                if isinstance(msg.content, str):
                    system_prompt = msg.content
                elif isinstance(msg.content, list):
                    # Extract text from content blocks
                    text_parts = []
                    for block in msg.content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                    system_prompt = " ".join(text_parts)

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
        self, content: str | list[dict[str, Any]] | None
    ) -> str | list[dict[str, Any]]:
        """Convert OpenAI content to Anthropic format."""
        if content is None:
            return ""

        if isinstance(content, str):
            return content

        # content must be a list at this point
        anthropic_content = []
        for block in content:
            if isinstance(block, dict):
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
        self, tools: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI tools to Anthropic format."""
        anthropic_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                anthropic_tools.append(
                    {
                        "name": func.get("name", ""),
                        "description": func.get("description", ""),
                        "input_schema": func.get("parameters", {}),
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
        }

        return mapping.get(stop_reason, "stop")
