"""Request conversion entry points for OpenAI→Anthropic adapters."""

from __future__ import annotations

import json
from typing import Any

from ccproxy.core.constants import DEFAULT_MAX_TOKENS
from ccproxy.core.logging import get_logger
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


logger = get_logger(__name__)


def _sanitize_tool_results(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove orphaned tool_result blocks that don't have matching tool_use blocks.

    The Anthropic API requires that each tool_result block must have a corresponding
    tool_use block in the immediately preceding assistant message. This function removes
    tool_result blocks that don't meet this requirement, converting them to text to
    preserve information.

    Args:
        messages: List of Anthropic format messages

    Returns:
        Sanitized messages with orphaned tool_results removed or converted to text
    """
    if not messages:
        return messages

    sanitized = []
    for i, msg in enumerate(messages):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            # Find tool_use_ids from the immediately preceding assistant message
            valid_tool_use_ids: set[str] = set()
            if i > 0 and messages[i - 1].get("role") == "assistant":
                prev_content = messages[i - 1].get("content", [])
                if isinstance(prev_content, list):
                    for block in prev_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_id = block.get("id")
                            if tool_id:
                                valid_tool_use_ids.add(tool_id)

            # Filter content blocks
            new_content = []
            orphaned_results = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id in valid_tool_use_ids:
                        new_content.append(block)
                    else:
                        # Track orphaned tool_result for conversion to text
                        orphaned_results.append(block)
                        logger.warning(
                            "orphaned_tool_result_removed",
                            tool_use_id=tool_use_id,
                            valid_ids=list(valid_tool_use_ids),
                            message_index=i,
                            category="message_sanitization",
                        )
                else:
                    new_content.append(block)

            # Convert orphaned results to text block to preserve information
            if orphaned_results:
                orphan_text = "[Previous tool results from compacted history]\n"
                for orphan in orphaned_results:
                    content = orphan.get("content", "")
                    if isinstance(content, list):
                        text_parts = []
                        for c in content:
                            if isinstance(c, dict) and c.get("type") == "text":
                                text_parts.append(c.get("text", ""))
                        content = "\n".join(text_parts)
                    # Truncate long content
                    content_str = str(content)
                    if len(content_str) > 500:
                        content_str = content_str[:500] + "..."
                    orphan_text += f"- Tool {orphan.get('tool_use_id', 'unknown')}: {content_str}\n"

                # Add as text block at the beginning
                new_content.insert(0, {"type": "text", "text": orphan_text})

            # Update message content (only if we have content left)
            if new_content:
                sanitized.append({**msg, "content": new_content})
            # If no content left, skip this message entirely
        else:
            sanitized.append(msg)

    return sanitized


async def convert__openai_chat_to_anthropic_message__request(
    request: openai_models.ChatCompletionRequest,
) -> anthropic_models.CreateMessageRequest:
    """Convert OpenAI ChatCompletionRequest to Anthropic CreateMessageRequest using typed models."""
    model = request.model.strip() if request.model else ""

    # Determine max tokens
    max_tokens = request.max_completion_tokens
    if max_tokens is None:
        # Access deprecated field with warning suppressed for backward compatibility
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            max_tokens = request.max_tokens
    if max_tokens is None:
        max_tokens = DEFAULT_MAX_TOKENS

    # Extract system message if present
    system_value: str | None = None
    out_messages: list[dict[str, Any]] = []

    for msg in request.messages or []:
        role = msg.role
        content = msg.content
        tool_calls = getattr(msg, "tool_calls", None)

        if role == "system":
            if isinstance(content, str):
                system_value = content
            elif isinstance(content, list):
                texts = [
                    part.text
                    for part in content
                    if hasattr(part, "type")
                    and part.type == "text"
                    and hasattr(part, "text")
                ]
                system_value = " ".join([t for t in texts if t]) or None
        elif role == "assistant":
            if tool_calls:
                blocks = []
                if content:  # Add text content if present
                    blocks.append({"type": "text", "text": str(content)})
                for tc in tool_calls:
                    func_info = tc.function
                    tool_name = func_info.name if func_info else None
                    tool_args = func_info.arguments if func_info else "{}"
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": str(tool_name) if tool_name is not None else "",
                            "input": json.loads(str(tool_args)),
                        }
                    )
                out_messages.append({"role": "assistant", "content": blocks})
            elif content is not None:
                out_messages.append({"role": "assistant", "content": content})

        elif role == "tool":
            tool_call_id = getattr(msg, "tool_call_id", None)
            out_messages.append(
                {
                    "role": "user",  # Anthropic uses 'user' role for tool results
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": str(content),
                        }
                    ],
                }
            )
        elif role == "user":
            if content is None:
                continue
            if isinstance(content, list):
                user_blocks: list[dict[str, Any]] = []
                text_accum: list[str] = []
                for part in content:
                    # Handle both dict and Pydantic object inputs
                    if isinstance(part, dict):
                        ptype = part.get("type")
                        if ptype == "text":
                            t = part.get("text")
                            if isinstance(t, str):
                                text_accum.append(t)
                        elif ptype == "image_url":
                            image_info = part.get("image_url")
                            if isinstance(image_info, dict):
                                url = image_info.get("url")
                                if isinstance(url, str) and url.startswith("data:"):
                                    try:
                                        header, b64data = url.split(",", 1)
                                        mediatype = header.split(";")[0].split(":", 1)[
                                            1
                                        ]
                                        user_blocks.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": str(mediatype),
                                                    "data": str(b64data),
                                                },
                                            }
                                        )
                                    except Exception:
                                        pass
                    elif hasattr(part, "type"):
                        # Pydantic object case
                        ptype = part.type
                        if ptype == "text" and hasattr(part, "text"):
                            t = part.text
                            if isinstance(t, str):
                                text_accum.append(t)
                        elif ptype == "image_url" and hasattr(part, "image_url"):
                            url = part.image_url.url if part.image_url else None
                            if isinstance(url, str) and url.startswith("data:"):
                                try:
                                    header, b64data = url.split(",", 1)
                                    mediatype = header.split(";")[0].split(":", 1)[1]
                                    user_blocks.append(
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": str(mediatype),
                                                "data": str(b64data),
                                            },
                                        }
                                    )
                                except Exception:
                                    pass
                if user_blocks:
                    # If we have images, always use list format
                    if text_accum:
                        user_blocks.insert(
                            0, {"type": "text", "text": " ".join(text_accum)}
                        )
                    out_messages.append({"role": "user", "content": user_blocks})
                elif len(text_accum) > 1:
                    # Multiple text parts - use list format
                    text_blocks = [{"type": "text", "text": " ".join(text_accum)}]
                    out_messages.append({"role": "user", "content": text_blocks})
                elif len(text_accum) == 1:
                    # Single text part - use string format
                    out_messages.append({"role": "user", "content": text_accum[0]})
                else:
                    # No content - use empty string
                    out_messages.append({"role": "user", "content": ""})
            else:
                out_messages.append({"role": "user", "content": content})

    # Sanitize tool_result blocks to ensure they have matching tool_use blocks
    out_messages = _sanitize_tool_results(out_messages)

    payload_data: dict[str, Any] = {
        "model": model,
        "messages": out_messages,
        "max_tokens": max_tokens,
    }

    # Inject system guidance for response_format JSON modes
    resp_fmt = request.response_format
    if resp_fmt is not None:
        inject: str | None = None
        if resp_fmt.type == "json_object":
            inject = (
                "Respond ONLY with a valid JSON object. "
                "Do not include any additional text, markdown, or explanation."
            )
        elif resp_fmt.type == "json_schema" and hasattr(resp_fmt, "json_schema"):
            schema = resp_fmt.json_schema
            try:
                if schema is not None:
                    schema_str = json.dumps(
                        schema.model_dump()
                        if hasattr(schema, "model_dump")
                        else schema,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                else:
                    schema_str = "{}"
            except Exception:
                schema_str = str(schema or {})
            inject = (
                "Respond ONLY with a JSON object that strictly conforms to this JSON Schema:\n"
                f"{schema_str}"
            )
        if inject:
            if system_value:
                system_value = f"{system_value}\n\n{inject}"
            else:
                system_value = inject

    if system_value is not None:
        # Ensure system value is a string, not a complex object
        if isinstance(system_value, str):
            payload_data["system"] = system_value
        else:
            # If system_value is not a string, try to extract text content
            try:
                if isinstance(system_value, list):
                    # Handle list format: [{"type": "text", "text": "...", "cache_control": {...}}]
                    text_parts = []
                    for part in system_value:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_content = part.get("text")
                            if isinstance(text_content, str):
                                text_parts.append(text_content)
                    if text_parts:
                        payload_data["system"] = " ".join(text_parts)
                elif (
                    isinstance(system_value, dict)
                    and system_value.get("type") == "text"
                ):
                    # Handle single dict format: {"type": "text", "text": "...", "cache_control": {...}}
                    text_content = system_value.get("text")
                    if isinstance(text_content, str):
                        payload_data["system"] = text_content
            except Exception:
                # Fallback: convert to string representation
                payload_data["system"] = str(system_value)
    if request.stream is not None:
        payload_data["stream"] = request.stream

    # Tools mapping (OpenAI function tools -> Anthropic tool definitions)
    tools_in = request.tools or []
    if tools_in:
        anth_tools: list[dict[str, Any]] = []
        for t in tools_in:
            if t.type == "function" and t.function is not None:
                fn = t.function
                anth_tools.append(
                    {
                        "type": "custom",
                        "name": fn.name,
                        "description": fn.description,
                        "input_schema": fn.parameters.model_dump()
                        if hasattr(fn.parameters, "model_dump")
                        else (fn.parameters or {}),
                    }
                )
        if anth_tools:
            payload_data["tools"] = anth_tools

    # tool_choice mapping
    tool_choice = request.tool_choice
    parallel_tool_calls = request.parallel_tool_calls
    disable_parallel = None
    if isinstance(parallel_tool_calls, bool):
        disable_parallel = not parallel_tool_calls

    if tool_choice is not None:
        anth_choice: dict[str, Any] | None = None
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                anth_choice = {"type": "none"}
            elif tool_choice == "auto":
                anth_choice = {"type": "auto"}
            elif tool_choice == "required":
                anth_choice = {"type": "any"}
        elif isinstance(tool_choice, dict):
            # Handle dict input like {"type": "function", "function": {"name": "search"}}
            if tool_choice.get("type") == "function" and isinstance(
                tool_choice.get("function"), dict
            ):
                anth_choice = {
                    "type": "tool",
                    "name": tool_choice["function"].get("name"),
                }
        elif hasattr(tool_choice, "type") and hasattr(tool_choice, "function"):
            # e.g., ChatCompletionNamedToolChoice pydantic model
            if tool_choice.type == "function" and tool_choice.function is not None:
                anth_choice = {
                    "type": "tool",
                    "name": tool_choice.function.name,
                }
        if anth_choice is not None:
            if disable_parallel is not None and anth_choice["type"] in {
                "auto",
                "any",
                "tool",
            }:
                anth_choice["disable_parallel_tool_use"] = disable_parallel
            payload_data["tool_choice"] = anth_choice

    # Thinking configuration
    thinking_cfg = derive_thinking_config(model, request)
    if thinking_cfg is not None:
        payload_data["thinking"] = thinking_cfg
        # Ensure token budget fits under max_tokens
        budget = thinking_cfg.get("budget_tokens", 0)
        if isinstance(budget, int) and max_tokens <= budget:
            payload_data["max_tokens"] = budget + 64
        # Temperature constraint when thinking enabled
        payload_data["temperature"] = 1.0

    # Validate against Anthropic model to ensure shape
    return anthropic_models.CreateMessageRequest.model_validate(payload_data)


def convert__openai_responses_to_anthropic_message__request(
    request: openai_models.ResponseRequest,
) -> anthropic_models.CreateMessageRequest:
    model = request.model
    stream = bool(request.stream)
    max_out = request.max_output_tokens

    messages: list[dict[str, Any]] = []
    system_parts: list[str] = []
    input_val = request.input

    if isinstance(input_val, str):
        messages.append({"role": "user", "content": input_val})
    elif isinstance(input_val, list):
        for item in input_val:
            if isinstance(item, dict) and item.get("type") == "message":
                role = item.get("role", "user")
                content_list = item.get("content", [])
                text_parts: list[str] = []
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") in {
                        "input_text",
                        "text",
                    }:
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                content_text = " ".join(text_parts)
                if role == "system":
                    system_parts.append(content_text)
                elif role in {"user", "assistant"}:
                    messages.append({"role": role, "content": content_text})
            elif hasattr(item, "type") and item.type == "message":
                role = getattr(item, "role", "user")
                content_list = getattr(item, "content", []) or []
                text_parts_alt: list[str] = []
                for part in content_list:
                    if hasattr(part, "type") and part.type in {"input_text", "text"}:
                        text = getattr(part, "text", None)
                        if isinstance(text, str):
                            text_parts_alt.append(text)
                content_text = " ".join(text_parts_alt)
                if role == "system":
                    system_parts.append(content_text)
                elif role in {"user", "assistant"}:
                    messages.append({"role": role, "content": content_text})

    payload_data: dict[str, Any] = {"model": model, "messages": messages}
    if max_out is None:
        max_out = DEFAULT_MAX_TOKENS
    payload_data["max_tokens"] = int(max_out)
    if stream:
        payload_data["stream"] = True

    if system_parts:
        payload_data["system"] = "\n".join(system_parts)

    tools_in = request.tools or []
    if tools_in:
        anth_tools: list[dict[str, Any]] = []
        for tool in tools_in:
            if isinstance(tool, dict):
                if tool.get("type") == "function":
                    fn = tool.get("function")
                    parameters = tool.get("parameters")
                    if isinstance(fn, dict):
                        name = fn.get("name") or tool.get("name")
                        description = fn.get("description") or tool.get("description")
                        schema = fn.get("parameters") or parameters or {}
                    else:
                        name = tool.get("name")
                        description = tool.get("description")
                        schema = parameters or {}

                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": name,
                            "description": description,
                            "input_schema": schema,
                        }
                    )
            elif (
                hasattr(tool, "type")
                and tool.type == "function"
                and hasattr(tool, "function")
                and tool.function is not None
            ):
                fn = tool.function
                anth_tools.append(
                    {
                        "type": "custom",
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": fn.parameters.model_dump()
                        if hasattr(fn.parameters, "model_dump")
                        else (fn.parameters or {}),
                    }
                )
        if anth_tools:
            payload_data["tools"] = anth_tools

    tool_choice = request.tool_choice
    parallel_tool_calls = request.parallel_tool_calls
    disable_parallel = None
    if isinstance(parallel_tool_calls, bool):
        disable_parallel = not parallel_tool_calls

    if tool_choice is not None:
        anth_choice: dict[str, Any] | None = None
        if isinstance(tool_choice, str):
            if tool_choice == "none":
                anth_choice = {"type": "none"}
            elif tool_choice == "auto":
                anth_choice = {"type": "auto"}
            elif tool_choice == "required":
                anth_choice = {"type": "any"}
        elif isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function" and isinstance(
                tool_choice.get("function"), dict
            ):
                anth_choice = {
                    "type": "tool",
                    "name": tool_choice["function"].get("name"),
                }
        elif hasattr(tool_choice, "type") and hasattr(tool_choice, "function"):
            if tool_choice.type == "function" and tool_choice.function is not None:
                anth_choice = {"type": "tool", "name": tool_choice.function.name}
        if anth_choice is not None:
            if disable_parallel is not None and anth_choice["type"] in {
                "auto",
                "any",
                "tool",
            }:
                anth_choice["disable_parallel_tool_use"] = disable_parallel
            payload_data["tool_choice"] = anth_choice

    text_cfg = request.text
    inject: str | None = None
    if text_cfg is not None:
        fmt = None
        if isinstance(text_cfg, dict):
            fmt = text_cfg.get("format")
        elif hasattr(text_cfg, "format"):
            fmt = text_cfg.format
        if fmt is not None:
            if isinstance(fmt, dict):
                fmt_type = fmt.get("type")
                if fmt_type == "json_schema":
                    schema = fmt.get("json_schema") or fmt.get("schema") or {}
                    try:
                        inject_schema = json.dumps(schema, separators=(",", ":"))
                    except Exception:
                        inject_schema = str(schema)
                    inject = (
                        "Respond ONLY with JSON strictly conforming to this JSON Schema:\n"
                        f"{inject_schema}"
                    )
                elif fmt_type == "json_object":
                    inject = (
                        "Respond ONLY with a valid JSON object. "
                        "No prose. Do not wrap in markdown."
                    )
            elif hasattr(fmt, "type"):
                if fmt.type == "json_object":
                    inject = (
                        "Respond ONLY with a valid JSON object. "
                        "No prose. Do not wrap in markdown."
                    )
                elif fmt.type == "json_schema" and (
                    hasattr(fmt, "json_schema") or hasattr(fmt, "schema")
                ):
                    schema_obj = getattr(fmt, "json_schema", None) or getattr(
                        fmt, "schema", None
                    )
                    try:
                        schema_data = (
                            schema_obj.model_dump()
                            if schema_obj and hasattr(schema_obj, "model_dump")
                            else schema_obj
                        )
                        inject_schema = json.dumps(schema_data, separators=(",", ":"))
                    except Exception:
                        inject_schema = str(schema_obj)
                    inject = (
                        "Respond ONLY with JSON strictly conforming to this JSON Schema:\n"
                        f"{inject_schema}"
                    )

    if inject:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{inject}" if existing_system else inject
        )

    text_instructions: str | None = None
    if isinstance(text_cfg, dict):
        text_instructions = text_cfg.get("instructions")
    elif text_cfg and hasattr(text_cfg, "instructions"):
        text_instructions = text_cfg.instructions

    if isinstance(text_instructions, str) and text_instructions:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{text_instructions}"
            if existing_system
            else text_instructions
        )

    if isinstance(request.instructions, str) and request.instructions:
        existing_system = payload_data.get("system")
        payload_data["system"] = (
            f"{existing_system}\n\n{request.instructions}"
            if existing_system
            else request.instructions
        )

    # Skip thinking config for ResponseRequest as it doesn't have the required fields
    thinking_cfg = None
    if thinking_cfg is not None:
        payload_data["thinking"] = thinking_cfg
        budget = thinking_cfg.get("budget_tokens", 0)
        if isinstance(budget, int) and payload_data.get("max_tokens", 0) <= budget:
            payload_data["max_tokens"] = budget + 64
        payload_data["temperature"] = 1.0

    return anthropic_models.CreateMessageRequest.model_validate(payload_data)


def derive_thinking_config(
    model: str, request: openai_models.ChatCompletionRequest
) -> dict[str, Any] | None:
    """Derive Anthropic thinking config from OpenAI fields and model name.

    Rules:
    - If model matches o1/o3 families, enable thinking by default with model-specific budget
    - Map reasoning_effort: low=1000, medium=5000, high=10000
    - o3*: 10000; o1-mini: 3000; other o1*: 5000
    - If thinking is enabled, return {"type":"enabled","budget_tokens":N}
    - Otherwise return None
    """
    # Explicit reasoning_effort mapping
    effort = getattr(request, "reasoning_effort", None)
    effort = effort.strip().lower() if isinstance(effort, str) else ""
    effort_budgets = {"low": 1000, "medium": 5000, "high": 10000}

    budget: int | None = None
    if effort in effort_budgets:
        budget = effort_budgets[effort]

    m = model.lower()
    # Model defaults if budget not set by effort
    if budget is None:
        if m.startswith("o3"):
            budget = 10000
        elif m.startswith("o1-mini"):
            budget = 3000
        elif m.startswith("o1"):
            budget = 5000

    if budget is None:
        return None

    return {"type": "enabled", "budget_tokens": budget}


__all__ = [
    "convert__openai_chat_to_anthropic_message__request",
    "convert__openai_responses_to_anthropic_message__request",
    "_sanitize_tool_results",  # Exposed for testing
]
