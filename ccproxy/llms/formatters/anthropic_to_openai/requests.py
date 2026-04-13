"""Request conversion entry points for Anthropic→OpenAI adapters."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ccproxy.llms.formatters.context import register_request
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


_MAX_CALL_ID_LEN = 64


def _clamp_call_id(call_id: Any) -> str | None:
    """Return a call_id that fits within OpenAI's 64-char limit.

    Deterministic: the same input always yields the same output, so a
    tool_use id and its matching tool_result.tool_use_id stay paired after
    clamping.
    """
    if not isinstance(call_id, str) or not call_id:
        return None
    if len(call_id) <= _MAX_CALL_ID_LEN:
        return call_id
    digest = hashlib.sha1(call_id.encode("utf-8")).hexdigest()
    return f"call_{digest}"


def _block_type(block: Any) -> Any:
    """Return ``block.type`` whether ``block`` is a dict or pydantic model."""
    if isinstance(block, dict):
        return block.get("type")
    return getattr(block, "type", None)


def _block_field(block: Any, name: str, default: Any = None) -> Any:
    """Return ``block[name]`` / ``block.name`` whether dict or pydantic model."""
    if isinstance(block, dict):
        return block.get(name, default)
    return getattr(block, name, default)


def _stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if _block_type(part) == "text":
                text = _block_field(part, "text")
                if isinstance(text, str):
                    text_parts.append(text)
                    continue
            try:
                text_parts.append(json.dumps(part, default=str))
            except Exception:
                text_parts.append(str(part))
        return "".join(text_parts)
    try:
        return json.dumps(content, default=str)
    except Exception:
        return str(content)


def _user_message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _assistant_message_item(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text}],
    }


def _function_call_item(block: Any) -> dict[str, Any]:
    tool_input = _block_field(block, "input") or {}
    try:
        args_str = json.dumps(tool_input)
    except Exception:
        args_str = json.dumps({"arguments": str(tool_input)})
    return {
        "type": "function_call",
        "call_id": _clamp_call_id(_block_field(block, "id")),
        "name": _block_field(block, "name"),
        "arguments": args_str,
    }


def _function_call_output_item(block: Any) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": _clamp_call_id(_block_field(block, "tool_use_id")),
        "output": _stringify_tool_result_content(_block_field(block, "content", "")),
    }


def _build_responses_input_items(
    messages: list[anthropic_models.Message],
) -> list[dict[str, Any]]:
    """Translate Anthropic messages into Responses API input items.

    Preserves the original order of text and tool_use/tool_result blocks
    inside each message. An assistant turn that interleaves text and
    tool_use blocks produces interleaved ``message`` and ``function_call``
    items, matching the Responses API's expectation that tool calls appear
    between the text segments that motivated them.
    """

    items: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, str):
            if msg.role == "assistant":
                items.append(_assistant_message_item(content))
            else:
                items.append(_user_message_item(content))
            continue

        if not isinstance(content, list):
            continue

        pending_text: list[str] = []
        make_message = (
            _assistant_message_item if msg.role == "assistant" else _user_message_item
        )

        for block in content:
            btype = _block_type(block)
            if btype == "text":
                text = _block_field(block, "text")
                if isinstance(text, str):
                    pending_text.append(text)
                continue
            if msg.role == "assistant" and btype == "tool_use":
                if pending_text:
                    items.append(make_message("".join(pending_text)))
                    pending_text = []
                items.append(_function_call_item(block))
                continue
            if msg.role == "user" and btype == "tool_result":
                if pending_text:
                    items.append(make_message("".join(pending_text)))
                    pending_text = []
                items.append(_function_call_output_item(block))
                continue

        if pending_text:
            items.append(make_message("".join(pending_text)))

    return items


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

    payload_data["input"] = _build_responses_input_items(request.messages)

    if request.tools:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if isinstance(
                tool, anthropic_models.Tool | anthropic_models.LegacyCustomTool
            ):
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
                                "id": getattr(block, "id", None),
                                "type": "function",
                                "function": {
                                    "name": getattr(block, "name", None),
                                    "arguments": args_str,
                                },
                            }
                        )
                elif block_type == "text":
                    btext = getattr(block, "text", None)
                    if isinstance(btext, str):
                        text_parts.append(btext)
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
                                    "tool_call_id": getattr(block, "tool_use_id", None),
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
                    btext_val = getattr(block, "text", None)
                    if btype == "text" and isinstance(btext_val, str):
                        text_accum.append(btext_val)
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
            if isinstance(
                tool, anthropic_models.Tool | anthropic_models.LegacyCustomTool
            ):
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
