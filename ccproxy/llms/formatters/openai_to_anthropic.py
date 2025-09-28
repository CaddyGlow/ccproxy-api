""" """

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from typing import Any, cast

from pydantic import BaseModel

from ccproxy.core.constants import DEFAULT_MAX_TOKENS
from ccproxy.llms.formatters.constants import OPENAI_TO_ANTHROPIC_ERROR_TYPE
from ccproxy.llms.formatters.utils import (
    map_openai_finish_to_anthropic_stop,
    openai_response_usage_snapshot,
    openai_usage_to_anthropic_usage,
    strict_parse_tool_arguments,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import OpenAIAccumulator


def convert__openai_to_anthropic__error(error: BaseModel) -> BaseModel:
    """Convert an OpenAI error payload to the Anthropic envelope."""
    if isinstance(error, openai_models.ErrorResponse):
        openai_error = error.error
        error_message = openai_error.message
        openai_error_type = openai_error.type or "api_error"
        anthropic_error_type = OPENAI_TO_ANTHROPIC_ERROR_TYPE.get(
            openai_error_type, "api_error"
        )

        anthropic_error: anthropic_models.ErrorType
        if anthropic_error_type == "invalid_request_error":
            anthropic_error = anthropic_models.InvalidRequestError(
                message=error_message
            )
        elif anthropic_error_type == "rate_limit_error":
            anthropic_error = anthropic_models.RateLimitError(message=error_message)
        else:
            anthropic_error = anthropic_models.APIError(message=error_message)

        return anthropic_models.ErrorResponse(error=anthropic_error)

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        fallback_error: anthropic_models.ErrorType = anthropic_models.APIError(
            message=error_message
        )
        return anthropic_models.ErrorResponse(error=fallback_error)

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        error_dict = error.model_dump()
        error_message = str(error_dict.get("message", error_dict))

    generic_error: anthropic_models.ErrorType = anthropic_models.APIError(
        message=error_message
    )
    return anthropic_models.ErrorResponse(error=generic_error)


THINKING_PATTERN = re.compile(
    r"<thinking(?:\s+signature=\"([^\"]*)\")?>(.*?)</thinking>",
    re.DOTALL,
)


def _to_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    return None


def _normalize_text_and_images(
    content: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    text_parts: list[str] = []
    image_blocks: list[dict[str, Any]] = []

    if not isinstance(content, list):
        return text_parts, image_blocks

    for part in content:
        mapping = _to_mapping(part)
        if not mapping:
            continue
        part_type = str(mapping.get("type", "")).lower()
        if part_type in {"text", "input_text"}:
            text_val = mapping.get("text")
            if isinstance(text_val, str) and text_val:
                text_parts.append(text_val)
        elif part_type == "image_url":
            image_info = mapping.get("image_url")
            image_map = _to_mapping(image_info)
            if not image_map:
                continue
            url = image_map.get("url")
            if isinstance(url, str) and url.startswith("data:"):
                try:
                    header, b64data = url.split(",", 1)
                    mediatype = header.split(";")[0].split(":", 1)[1]
                except (ValueError, IndexError):
                    continue
                image_blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mediatype,
                            "data": b64data,
                        },
                    }
                )

    return text_parts, image_blocks


def _stringify_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts, _ = _normalize_text_and_images(content)
        return " ".join(text_parts)
    return str(content)


def _coerce_system_content(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts, _ = _normalize_text_and_images(content)
        return " ".join(text_parts) if text_parts else None
    return None


def _build_user_blocks(content: Any) -> str | list[dict[str, Any]] | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content

    text_parts, image_blocks = _normalize_text_and_images(content)

    if not text_parts and not image_blocks:
        return ""

    blocks: list[dict[str, Any]] = []
    if text_parts:
        blocks.append({"type": "text", "text": " ".join(text_parts)})
    blocks.extend(image_blocks)

    if len(blocks) == 1 and blocks[0]["type"] == "text":
        return str(blocks[0]["text"])
    return blocks


def _build_assistant_blocks(
    content: Any, tool_calls: list[openai_models.ToolCall] | None
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []

    normalized_content = _build_user_blocks(content)
    if isinstance(normalized_content, str):
        text = normalized_content.strip()
        if text:
            blocks.append({"type": "text", "text": text})
    elif isinstance(normalized_content, list):
        blocks.extend(normalized_content)

    for call in tool_calls or []:
        args_dict = strict_parse_tool_arguments(call.function.arguments)
        blocks.append(
            {
                "type": "tool_use",
                "id": call.id,
                "name": call.function.name,
                "input": args_dict,
            }
        )

    return blocks


def convert__openai_responses_usage_to_openai_completion__usage(
    usage: openai_models.ResponseUsage,
) -> openai_models.CompletionUsage:
    snapshot = openai_response_usage_snapshot(usage)

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=snapshot.cache_read_tokens, audio_tokens=0
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=snapshot.reasoning_tokens,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return openai_models.CompletionUsage(
        prompt_tokens=snapshot.input_tokens,
        completion_tokens=snapshot.output_tokens,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert__openai_responses_usage_to_anthropic__usage(
    usage: openai_models.ResponseUsage,
) -> anthropic_models.Usage:
    snapshot = openai_response_usage_snapshot(usage)

    return anthropic_models.Usage(
        input_tokens=snapshot.input_tokens,
        output_tokens=snapshot.output_tokens,
        cache_read_input_tokens=snapshot.cache_read_tokens,
        cache_creation_input_tokens=0,
    )


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


def convert__openai_responses_to_anthropic_message__response(
    response: openai_models.ResponseObject,
) -> anthropic_models.MessageResponse:
    from ccproxy.llms.models.anthropic import (
        TextBlock as AnthropicTextBlock,
    )
    from ccproxy.llms.models.anthropic import (
        ThinkingBlock as AnthropicThinkingBlock,
    )
    from ccproxy.llms.models.anthropic import (
        ToolUseBlock as AnthropicToolUseBlock,
    )

    content_blocks: list[
        AnthropicTextBlock | AnthropicThinkingBlock | AnthropicToolUseBlock
    ] = []

    for item in response.output or []:
        item_type = getattr(item, "type", None)
        if item_type == "reasoning":
            summary_parts = getattr(item, "summary", []) or []
            texts: list[str] = []
            for part in summary_parts:
                part_type = getattr(part, "type", None)
                if part_type == "summary_text":
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        texts.append(text)
            if texts:
                content_blocks.append(
                    AnthropicThinkingBlock(
                        type="thinking",
                        thinking=" ".join(texts),
                        signature="",
                    )
                )
        elif item_type == "function_call":
            raw_name = getattr(item, "name", None)
            call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
            raw_arguments = getattr(item, "arguments", None)

            try:
                input_payload = strict_parse_tool_arguments(raw_arguments)
            except Exception:
                if isinstance(raw_arguments, str | bytes):
                    input_payload = {"arguments": raw_arguments}
                else:
                    input_payload = {"arguments": raw_arguments or ""}

            content_blocks.append(
                AnthropicToolUseBlock(
                    type="tool_use",
                    id=call_id or f"tool_{len(content_blocks)}",
                    name=str(raw_name or "function"),
                    input=input_payload,
                )
            )

    for item in response.output or []:
        item_type = getattr(item, "type", None)
        if item_type == "message":
            content_list = getattr(item, "content", []) or []
            for part in content_list:
                if hasattr(part, "type") and part.type == "output_text":
                    text = getattr(part, "text", "") or ""
                    last_idx = 0
                    for match in THINKING_PATTERN.finditer(text):
                        if match.start() > last_idx:
                            prefix = text[last_idx : match.start()]
                            if prefix.strip():
                                content_blocks.append(
                                    AnthropicTextBlock(type="text", text=prefix)
                                )
                        signature = match.group(1) or ""
                        thinking_text = match.group(2) or ""
                        content_blocks.append(
                            AnthropicThinkingBlock(
                                type="thinking",
                                thinking=thinking_text,
                                signature=signature,
                            )
                        )
                        last_idx = match.end()
                    tail = text[last_idx:]
                    if tail.strip():
                        content_blocks.append(
                            AnthropicTextBlock(type="text", text=tail)
                        )
                elif isinstance(part, dict):
                    part_type = part.get("type")
                    if part_type == "output_text":
                        text = part.get("text", "") or ""
                        last_idx = 0
                        for match in THINKING_PATTERN.finditer(text):
                            if match.start() > last_idx:
                                prefix = text[last_idx : match.start()]
                                if prefix.strip():
                                    content_blocks.append(
                                        AnthropicTextBlock(type="text", text=prefix)
                                    )
                            signature = match.group(1) or ""
                            thinking_text = match.group(2) or ""
                            content_blocks.append(
                                AnthropicThinkingBlock(
                                    type="thinking",
                                    thinking=thinking_text,
                                    signature=signature,
                                )
                            )
                            last_idx = match.end()
                        tail = text[last_idx:]
                        if tail.strip():
                            content_blocks.append(
                                AnthropicTextBlock(type="text", text=tail)
                            )
                    elif part_type == "tool_use":
                        content_blocks.append(
                            AnthropicToolUseBlock(
                                type="tool_use",
                                id=part.get("id", "tool_1"),
                                name=part.get("name", "function"),
                                input=part.get("arguments", part.get("input", {}))
                                or {},
                            )
                        )
                elif (
                    hasattr(part, "type") and getattr(part, "type", None) == "tool_use"
                ):
                    content_blocks.append(
                        AnthropicToolUseBlock(
                            type="tool_use",
                            id=getattr(part, "id", "tool_1") or "tool_1",
                            name=getattr(part, "name", "function") or "function",
                            input=getattr(part, "arguments", getattr(part, "input", {}))
                            or {},
                        )
                    )

    usage = openai_usage_to_anthropic_usage(response.usage)

    return anthropic_models.MessageResponse(
        id=response.id or "msg_1",
        type="message",
        role="assistant",
        model=response.model or "",
        content=cast(list[anthropic_models.ResponseContentBlock], content_blocks),
        stop_reason="end_turn",
        stop_sequence=None,
        usage=usage,
    )


async def convert__openai_responses_to_anthropic_messages__stream(
    stream: AsyncIterator[Any],
) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
    """Translate OpenAI Responses streaming events into Anthropic message events."""

    def _event_to_dict(raw: Any) -> dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "root"):
            return _event_to_dict(raw.root)
        if hasattr(raw, "model_dump"):
            return cast(dict[str, Any], raw.model_dump(mode="json"))
        return cast(dict[str, Any], {})

    def _parse_tool_input(text: str) -> dict[str, Any]:
        if not text:
            return cast(dict[str, Any], {})
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {"arguments": text}
        except Exception:
            return {"arguments": text}

    message_started = False
    text_block_active = False
    current_index = 0
    final_stop_reason: str | None = None
    final_stop_sequence: str | None = None
    usage = anthropic_models.Usage(input_tokens=0, output_tokens=0)
    reasoning_buffer: list[str] = []
    tool_args: dict[int, list[str]] = {}
    tool_meta: dict[int, dict[str, Any]] = {}

    async for raw_event in stream:
        event = _event_to_dict(raw_event)
        event_type = event.get("type") or event.get("event")
        if not event_type:
            continue

        if event_type == "error":
            payload = event.get("error") or {}
            detail = (
                anthropic_models.ErrorDetail(**payload)
                if isinstance(payload, dict)
                else anthropic_models.ErrorDetail(message=str(payload))
            )
            yield anthropic_models.ErrorEvent(type="error", error=detail)
            return

        if not message_started:
            response_meta = event.get("response") or {}
            yield anthropic_models.MessageStartEvent(
                type="message_start",
                message=anthropic_models.MessageResponse(
                    id=response_meta.get("id", "resp_stream"),
                    type="message",
                    role="assistant",
                    content=[],
                    model=response_meta.get("model", ""),
                    stop_reason=None,
                    stop_sequence=None,
                    usage=usage,
                ),
            )
            message_started = True

        if event_type == "response.output_text.delta":
            delta = event.get("delta")
            text = ""
            if isinstance(delta, dict):
                text = delta.get("text") or ""
            elif isinstance(delta, str):
                text = delta
            if text:
                if not text_block_active:
                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=current_index,
                        content_block=anthropic_models.TextBlock(type="text", text=""),
                    )
                    text_block_active = True
                yield anthropic_models.ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=current_index,
                    delta=anthropic_models.TextBlock(type="text", text=text),
                )
        elif event_type == "response.output_text.done":
            if text_block_active:
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                text_block_active = False
                current_index += 1

        elif event_type == "response.reasoning_summary_text.delta":
            delta = event.get("delta")
            summary_piece = delta.get("text") if isinstance(delta, dict) else delta
            if isinstance(summary_piece, str):
                reasoning_buffer.append(summary_piece)

        elif event_type == "response.reasoning_summary_text.done":
            if text_block_active:
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                text_block_active = False
                current_index += 1
            summary = "".join(reasoning_buffer)
            reasoning_buffer.clear()
            if summary:
                yield anthropic_models.ContentBlockStartEvent(
                    type="content_block_start",
                    index=current_index,
                    content_block=anthropic_models.ThinkingBlock(
                        type="thinking",
                        thinking=summary,
                        signature="",
                    ),
                )
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                current_index += 1

        elif event_type == "response.function_call_arguments.delta":
            output_index = event.get("output_index", 0)
            delta = event.get("delta") or {}
            delta_text = delta.get("arguments") if isinstance(delta, dict) else delta
            if isinstance(delta_text, str):
                tool_args.setdefault(output_index, []).append(delta_text)

            meta = tool_meta.setdefault(
                output_index,
                {
                    "id": event.get("item_id", f"call_{output_index}"),
                },
            )
            name_val = event.get("name")
            if isinstance(name_val, str) and name_val:
                meta["name"] = name_val
            call_id_val = event.get("call_id")
            if isinstance(call_id_val, str) and call_id_val:
                meta["call_id"] = call_id_val

        elif event_type == "response.function_call_arguments.done":
            output_index = event.get("output_index", 0)
            arguments = event.get("arguments") if isinstance(event, dict) else None
            args_buffer = tool_args.get(output_index, [])
            args_str = arguments if isinstance(arguments, str) else "".join(args_buffer)

            meta = tool_meta.setdefault(
                output_index,
                {
                    "id": event.get("item_id", f"call_{output_index}"),
                },
            )
            if args_str:
                meta["arguments"] = args_str
            name_val = event.get("name")
            if isinstance(name_val, str) and name_val:
                meta["name"] = name_val
            call_id_val = event.get("call_id")
            if isinstance(call_id_val, str) and call_id_val:
                meta["call_id"] = call_id_val

        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "output_tool_call":
                output_index = item.get("output_index", event.get("output_index", 0))
                meta = tool_meta.setdefault(output_index, {})
                meta["id"] = item.get("id", f"call_{output_index}")
                if item.get("name"):
                    meta["name"] = item.get("name")
                if item.get("call_id"):
                    meta["call_id"] = item.get("call_id")
                tool_args.setdefault(output_index, [])

        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "output_text" and text_block_active:
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                text_block_active = False
                current_index += 1
            elif item_type == "output_tool_call":
                output_index = item.get("output_index", event.get("output_index", 0))
                args_buffer = tool_args.pop(output_index, [])
                meta = tool_meta.pop(
                    output_index,
                    {
                        "id": item.get("id", f"call_{output_index}"),
                    },
                )
                item_name = item.get("name")
                if isinstance(item_name, str) and item_name:
                    meta["name"] = item_name
                item_call_id = item.get("call_id")
                if isinstance(item_call_id, str) and item_call_id:
                    meta["call_id"] = item_call_id
                if item.get("id") and not meta.get("id"):
                    meta["id"] = item.get("id")

                args_str = meta.pop("arguments", None)
                if not isinstance(args_str, str) or not args_str:
                    args_str = "".join(args_buffer)

                yield anthropic_models.ContentBlockStartEvent(
                    type="content_block_start",
                    index=current_index,
                    content_block=anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id=meta.get("id", f"call_{output_index}"),
                        name=meta.get("name") or "tool",
                        input=_parse_tool_input(args_str),
                    ),
                )
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                current_index += 1

        elif event_type == "response.completed":
            response = event.get("response") or {}
            usage_data = response.get("usage") or {}
            try:
                usage = anthropic_models.Usage.model_validate(usage_data)
            except Exception:
                usage = anthropic_models.Usage(
                    input_tokens=usage_data.get("input_tokens", 0),
                    output_tokens=usage_data.get("output_tokens", 0),
                )
            final_stop_reason = response.get("stop_reason")
            final_stop_sequence = response.get("stop_sequence")
            break

    if text_block_active:
        yield anthropic_models.ContentBlockStopEvent(
            type="content_block_stop", index=current_index
        )

    if message_started:
        yield anthropic_models.MessageDeltaEvent(
            type="message_delta",
            delta=anthropic_models.MessageDelta(
                stop_reason=map_openai_finish_to_anthropic_stop(final_stop_reason),
                stop_sequence=final_stop_sequence,
            ),
            usage=usage,
        )
        yield anthropic_models.MessageStopEvent(type="message_stop")


def convert__openai_chat_to_anthropic_messages__stream(
    stream: AsyncIterator[openai_models.ChatCompletionChunk],
) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
    """Convert OpenAI ChatCompletion stream to Anthropic MessageStreamEvent stream."""

    async def generator() -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        message_started = False
        text_block_started = False
        accumulated_content = ""
        model_id = ""
        current_index = 0
        emitted_tool_ids: set[str] = set()
        openai_accumulator = OpenAIAccumulator()

        async for chunk in stream:
            if isinstance(chunk, dict):
                chunk_payload = chunk
            else:
                try:
                    chunk_payload = chunk.model_dump(mode="json", exclude_none=True)
                except Exception:
                    chunk_payload = chunk.model_dump(exclude_none=True)

            choices = chunk_payload.get("choices")
            if not choices:
                continue

            choice = choices[0]
            openai_accumulator.accumulate("", chunk_payload)

            if chunk_payload.get("model"):
                model_id = chunk_payload["model"]
            elif not isinstance(chunk, dict):
                model_id = getattr(chunk, "model", model_id) or model_id

            delta = choice.get("delta") if isinstance(choice, dict) else {}
            if not isinstance(delta, dict):
                delta = {}
            finish_reason = choice.get("finish_reason")
            content_piece = delta.get("content")

            if not message_started:
                chunk_id = chunk_payload.get("id")
                if chunk_id is None and not isinstance(chunk, dict):
                    chunk_id = getattr(chunk, "id", None)
                chunk_id = chunk_id or "msg_stream"

                yield anthropic_models.MessageStartEvent(
                    type="message_start",
                    message=anthropic_models.MessageResponse(
                        id=chunk_id,
                        type="message",
                        role="assistant",
                        content=[],
                        model=model_id,
                        stop_reason=None,
                        stop_sequence=None,
                        usage=anthropic_models.Usage(input_tokens=0, output_tokens=0),
                    ),
                )
                message_started = True

            if content_piece:
                content_text = (
                    content_piece
                    if isinstance(content_piece, str)
                    else str(content_piece)
                )
                if not text_block_started:
                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=current_index,
                        content_block=anthropic_models.TextBlock(type="text", text=""),
                    )
                    text_block_started = True

                yield anthropic_models.ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=current_index,
                    delta=anthropic_models.TextBlock(type="text", text=content_text),
                )
                accumulated_content += content_text

            if finish_reason:
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop",
                        index=current_index,
                    )
                    text_block_started = False
                    current_index += 1

                for tool in openai_accumulator.get_complete_tool_calls():
                    tool_key = (
                        tool.get("id")
                        or f"call_{tool.get('index', len(emitted_tool_ids))}"
                    )
                    tool_key_str = str(tool_key)
                    if tool_key_str in emitted_tool_ids:
                        continue

                    raw_arguments = tool.get("function", {}).get("arguments", "")
                    try:
                        arguments = strict_parse_tool_arguments(raw_arguments)
                    except Exception:
                        arguments = {"arguments": raw_arguments}

                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=current_index,
                        content_block=anthropic_models.ToolUseBlock(
                            type="tool_use",
                            id=tool_key_str,
                            name=str(
                                tool.get("function", {}).get("name")
                                or tool.get("name")
                                or "function"
                            ),
                            input=arguments,
                        ),
                    )
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=current_index
                    )
                    emitted_tool_ids.add(tool_key_str)
                    current_index += 1

                usage_payload = chunk_payload.get("usage")
                if usage_payload is None and not isinstance(chunk, dict):
                    usage_obj = getattr(chunk, "usage", None)
                    if usage_obj is not None:
                        if hasattr(usage_obj, "model_dump"):
                            usage_payload = usage_obj.model_dump()
                        else:
                            usage_payload = {
                                "prompt_tokens": getattr(usage_obj, "prompt_tokens", 0),
                                "completion_tokens": getattr(
                                    usage_obj, "completion_tokens", 0
                                ),
                            }

                anthropic_usage = (
                    openai_usage_to_anthropic_usage(usage_payload)
                    if usage_payload is not None
                    else anthropic_models.Usage(input_tokens=0, output_tokens=0)
                )

                mapped_stop = map_openai_finish_to_anthropic_stop(finish_reason)

                yield anthropic_models.MessageDeltaEvent(
                    type="message_delta",
                    delta=anthropic_models.MessageDelta(stop_reason=mapped_stop),
                    usage=anthropic_usage,
                )
                yield anthropic_models.MessageStopEvent(type="message_stop")
                break

    return generator()


def convert__openai_chat_to_anthropic_messages__response(
    response: openai_models.ChatCompletionResponse,
) -> anthropic_models.MessageResponse:
    """Convert OpenAI ChatCompletionResponse to Anthropic MessageResponse."""
    text_content = ""
    finish_reason = None
    tool_contents: list[anthropic_models.ToolUseBlock] = []
    if response.choices:
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        msg = getattr(choice, "message", None)
        if msg is not None:
            content_val = getattr(msg, "content", None)
            if isinstance(content_val, str):
                text_content = content_val
            elif isinstance(content_val, list):
                parts: list[str] = []
                for part in content_val:
                    if isinstance(part, dict) and part.get("type") == "text":
                        t = part.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                text_content = "".join(parts)

            # Extract OpenAI Chat tool calls (strict JSON parsing)
            tool_calls = getattr(msg, "tool_calls", None)
            if isinstance(tool_calls, list):
                for i, tc in enumerate(tool_calls):
                    fn = getattr(tc, "function", None)
                    if fn is None and isinstance(tc, dict):
                        fn = tc.get("function")
                    if not fn:
                        continue
                    name = getattr(fn, "name", None)
                    if name is None and isinstance(fn, dict):
                        name = fn.get("name")
                    args_raw = getattr(fn, "arguments", None)
                    if args_raw is None and isinstance(fn, dict):
                        args_raw = fn.get("arguments")
                    args = strict_parse_tool_arguments(args_raw)
                    tool_id = getattr(tc, "id", None)
                    if tool_id is None and isinstance(tc, dict):
                        tool_id = tc.get("id")
                    tool_contents.append(
                        anthropic_models.ToolUseBlock(
                            type="tool_use",
                            id=tool_id or f"call_{i}",
                            name=name or "function",
                            input=args,
                        )
                    )
            # Legacy single function
            legacy_fn = getattr(msg, "function", None)
            if legacy_fn:
                name = getattr(legacy_fn, "name", None)
                args_raw = getattr(legacy_fn, "arguments", None)
                args = strict_parse_tool_arguments(args_raw)
                tool_contents.append(
                    anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id="call_0",
                        name=name or "function",
                        input=args,
                    )
                )

    content_blocks: list[anthropic_models.ResponseContentBlock] = []
    if text_content:
        content_blocks.append(
            anthropic_models.TextBlock(type="text", text=text_content)
        )
    # Append tool blocks after text (order matches Responses path patterns)
    content_blocks.extend(tool_contents)

    # Map usage via shared utility
    usage = openai_usage_to_anthropic_usage(getattr(response, "usage", None))

    stop_reason = map_openai_finish_to_anthropic_stop(finish_reason)

    return anthropic_models.MessageResponse(
        id=getattr(response, "id", "msg_1") or "msg_1",
        type="message",
        role="assistant",
        model=getattr(response, "model", "") or "",
        content=content_blocks,
        stop_reason=stop_reason,
        stop_sequence=None,
        usage=usage,
    )
