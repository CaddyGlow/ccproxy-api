"""Request conversion entry points for Anthropic→OpenAI adapters."""

from __future__ import annotations

import json
from typing import Any

from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


def _build_responses_payload_from_anthropic_request(
    request: anthropic_models.CreateMessageRequest,
) -> tuple[dict[str, Any], str | None]:
    """Project an Anthropic message request into Responses payload fields."""

    payload_data: dict[str, Any] = {"model": request.model}
    instructions_text: str | None = None

    if request.max_tokens is not None:
        payload_data["max_output_tokens"] = int(request.max_tokens)
    if request.stream:
        payload_data["stream"] = True

    if request.service_tier is not None:
        payload_data["service_tier"] = request.service_tier
    if request.temperature is not None:
        payload_data["temperature"] = request.temperature
    if request.top_p is not None:
        payload_data["top_p"] = request.top_p

    if request.metadata is not None and hasattr(request.metadata, "model_dump"):
        meta_dump = request.metadata.model_dump()
        payload_data["metadata"] = meta_dump

    if request.system:
        if isinstance(request.system, str):
            instructions_text = request.system
            payload_data["instructions"] = request.system
        else:
            joined = "".join(block.text for block in request.system if block.text)
            instructions_text = joined or None
            if joined:
                payload_data["instructions"] = joined

    last_user_text: str | None = None
    for msg in reversed(request.messages):
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            last_user_text = msg.content
        elif isinstance(msg.content, list):
            texts: list[str] = []
            for block in msg.content:
                if isinstance(block, dict):
                    if block.get("type") == "text" and isinstance(
                        block.get("text"), str
                    ):
                        texts.append(block.get("text") or "")
                elif (
                    getattr(block, "type", None) == "text"
                    and hasattr(block, "text")
                    and isinstance(getattr(block, "text", None), str)
                ):
                    texts.append(block.text or "")
            if texts:
                last_user_text = " ".join(texts)
        break

    if last_user_text:
        payload_data["input"] = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": last_user_text},
                ],
            }
        ]
    else:
        payload_data["input"] = []

    if request.tools:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if isinstance(tool, anthropic_models.Tool):
                tools.append(
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    }
                )
        if tools:
            payload_data["tools"] = tools

    tc = request.tool_choice
    if tc is not None:
        tc_type = getattr(tc, "type", None)
        if tc_type == "none":
            payload_data["tool_choice"] = "none"
        elif tc_type == "auto":
            payload_data["tool_choice"] = "auto"
        elif tc_type == "any":
            payload_data["tool_choice"] = "required"
        elif tc_type == "tool":
            name = getattr(tc, "name", None)
            if name:
                payload_data["tool_choice"] = {
                    "type": "function",
                    "function": {"name": name},
                }
        disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
        if isinstance(disable_parallel, bool):
            payload_data["parallel_tool_calls"] = not disable_parallel

    payload_data.setdefault("background", None)

    return payload_data, instructions_text


def convert__anthropic_message_to_openai_responses__request(
    request: anthropic_models.CreateMessageRequest,
) -> openai_models.ResponseRequest:
    """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest using typed models."""
    payload_data, instructions_text = _build_responses_payload_from_anthropic_request(
        request
    )

    response_request = openai_models.ResponseRequest.model_validate(payload_data)

    register_request(request, instructions_text)

    return response_request


def convert__anthropic_message_to_openai_chat__request(
    request: anthropic_models.CreateMessageRequest,
) -> openai_models.ChatCompletionRequest:
    """Convert Anthropic CreateMessageRequest to OpenAI ChatCompletionRequest using typed models."""
    openai_messages: list[dict[str, Any]] = []
    # System prompt
    if request.system:
        if isinstance(request.system, str):
            sys_content = request.system
        else:
            sys_content = "".join(block.text for block in request.system)
        if sys_content:
            openai_messages.append({"role": "system", "content": sys_content})

    # User/assistant messages with text + data-url images
    for msg in request.messages:
        role = msg.role
        content = msg.content

        # Handle tool usage and results
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            text_parts = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    # Type guard for ToolUseBlock
                    if hasattr(block, "id") and hasattr(block, "name"):
                        # Safely get input with fallback to empty dict
                        tool_input = getattr(block, "input", {}) or {}

                        # Ensure input is properly serialized as JSON
                        try:
                            args_str = json.dumps(tool_input)
                        except Exception:
                            args_str = json.dumps({"arguments": str(tool_input)})

                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": args_str,
                                },
                            }
                        )
                elif block_type == "text":
                    # Type guard for TextBlock
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
            if tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                assistant_msg["content"] = " ".join(text_parts) if text_parts else None
                openai_messages.append(assistant_msg)
                continue
        elif role == "user" and isinstance(content, list):
            is_tool_result = any(
                getattr(b, "type", None) == "tool_result" for b in content
            )
            if is_tool_result:
                for block in content:
                    if getattr(block, "type", None) == "tool_result":
                        # Type guard for ToolResultBlock
                        if hasattr(block, "tool_use_id"):
                            # Get content with an empty string fallback
                            result_content = getattr(block, "content", "")

                            # Convert complex content to string representation
                            if not isinstance(result_content, str):
                                try:
                                    if isinstance(result_content, list):
                                        # Handle list of text blocks
                                        text_parts = []
                                        for part in result_content:
                                            if (
                                                hasattr(part, "text")
                                                and hasattr(part, "type")
                                                and part.type == "text"
                                            ):
                                                text_parts.append(part.text)
                                        if text_parts:
                                            result_content = " ".join(text_parts)
                                        else:
                                            result_content = json.dumps(result_content)
                                    else:
                                        # Convert other non-string content to JSON
                                        result_content = json.dumps(result_content)
                                except Exception:
                                    # Fallback to string representation
                                    result_content = str(result_content)

                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": result_content,
                                }
                            )
                continue

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            text_accum: list[str] = []
            for block in content:
                # Support both raw dicts and Anthropic model instances
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and isinstance(block.get("text"), str):
                        text_accum.append(block.get("text") or "")
                    elif btype == "image":
                        source = block.get("source") or {}
                        if (
                            isinstance(source, dict)
                            and source.get("type") == "base64"
                            and isinstance(source.get("media_type"), str)
                            and isinstance(source.get("data"), str)
                        ):
                            url = f"data:{source['media_type']};base64,{source['data']}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
                else:
                    # Pydantic models
                    btype = getattr(block, "type", None)
                    if (
                        btype == "text"
                        and hasattr(block, "text")
                        and isinstance(getattr(block, "text", None), str)
                    ):
                        text_accum.append(block.text or "")
                    elif btype == "image":
                        source = getattr(block, "source", None)
                        if (
                            source is not None
                            and getattr(source, "type", None) == "base64"
                            and isinstance(getattr(source, "media_type", None), str)
                            and isinstance(getattr(source, "data", None), str)
                        ):
                            url = f"data:{source.media_type};base64,{source.data}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
            if parts or len(text_accum) > 1:
                if text_accum:
                    parts.insert(0, {"type": "text", "text": " ".join(text_accum)})
                openai_messages.append({"role": role, "content": parts})
            else:
                openai_messages.append(
                    {"role": role, "content": (text_accum[0] if text_accum else "")}
                )
        else:
            openai_messages.append({"role": role, "content": content})

    # Tools mapping (custom tools -> function tools)
    tools: list[dict[str, Any]] = []
    if request.tools:
        for tool in request.tools:
            if isinstance(tool, anthropic_models.Tool):
                tools.append(
                    {
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        },
                    }
                )

    params: dict[str, Any] = {
        "model": request.model,
        "messages": openai_messages,
        "max_completion_tokens": request.max_tokens,
        "stream": request.stream or None,
    }
    if tools:
        params["tools"] = tools

    # tool_choice mapping
    tc = request.tool_choice
    if tc is not None:
        tc_type = getattr(tc, "type", None)
        if tc_type == "none":
            params["tool_choice"] = "none"
        elif tc_type == "auto":
            params["tool_choice"] = "auto"
        elif tc_type == "any":
            params["tool_choice"] = "required"
        elif tc_type == "tool":
            name = getattr(tc, "name", None)
            if name:
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": name},
                }
        # parallel_tool_calls from disable_parallel_tool_use
        disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
        if isinstance(disable_parallel, bool):
            params["parallel_tool_calls"] = not disable_parallel

    # Validate against OpenAI model
    return openai_models.ChatCompletionRequest.model_validate(params)


__all__ = [
    "convert__anthropic_message_to_openai_chat__request",
    "convert__anthropic_message_to_openai_responses__request",
]
