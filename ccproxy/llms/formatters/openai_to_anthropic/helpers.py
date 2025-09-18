""" """

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

from pydantic import BaseModel

from ccproxy.core.constants import DEFAULT_MAX_TOKENS
from ccproxy.llms.formatters.shared.constants import (
    OPENAI_TO_ANTHROPIC_ERROR_TYPE,
)
from ccproxy.llms.formatters.shared.utils import (
    map_openai_finish_to_anthropic_stop,
    openai_usage_to_anthropic_usage,
    strict_parse_tool_arguments,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


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


def convert__openai_responses_usage_to_openai_completion__usage(
    usage: openai_models.ResponseUsage,
) -> openai_models.CompletionUsage:
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details:
        cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

    reasoning_tokens = 0
    output_details = getattr(usage, "output_tokens_details", None)
    if output_details:
        reasoning_tokens = int(getattr(output_details, "reasoning_tokens", 0) or 0)

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=cached_tokens, audio_tokens=0
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=reasoning_tokens,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return openai_models.CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert__openai_responses_usage_to_anthropic__usage(
    usage: openai_models.ResponseUsage,
) -> anthropic_models.Usage:
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    # Extract cache information if available
    cache_read_tokens = 0
    cache_creation_tokens = 0
    input_details = getattr(usage, "input_tokens_details", None)
    if input_details:
        cache_read_tokens = int(getattr(input_details, "cached_tokens", 0) or 0)

    return anthropic_models.Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=cache_read_tokens,
        cache_creation_input_tokens=cache_creation_tokens,
    )


async def convert__openai_chat_to_anthropic_message__request(
    request: openai_models.ChatCompletionRequest,
) -> anthropic_models.CreateMessageRequest:
    """Convert OpenAI ChatCompletionRequest to Anthropic CreateMessageRequest using typed models."""
    model = request.model.strip() if request.model else ""

    # Determine max tokens
    max_tokens = request.max_completion_tokens
    if max_tokens is None:
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

    # Tools mapping (OpenAI function tools -> Anthropic custom tools)
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
                if tool.get("type") == "function" and isinstance(
                    tool.get("function"), dict
                ):
                    fn = tool["function"]
                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": fn.get("name"),
                            "description": fn.get("description"),
                            "input_schema": fn.get("parameters") or {},
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
                        "name": fn.name,
                        "description": fn.description,
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
    tool_meta: dict[int, dict[str, str]] = {}

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
            tool_meta.setdefault(
                output_index,
                {
                    "id": event.get("item_id", f"call_{output_index}"),
                    "name": event.get("name", "tool"),
                },
            )

        elif event_type == "response.function_call_arguments.done":
            output_index = event.get("output_index", 0)
            args = "".join(tool_args.pop(output_index, []))
            meta = tool_meta.pop(
                output_index,
                {
                    "id": event.get("item_id", f"call_{output_index}"),
                    "name": event.get("name", "tool"),
                },
            )
            if text_block_active:
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=current_index
                )
                text_block_active = False
                current_index += 1
            yield anthropic_models.ContentBlockStartEvent(
                type="content_block_start",
                index=current_index,
                content_block=anthropic_models.ToolUseBlock(
                    type="tool_use",
                    id=meta.get("id", f"call_{output_index}"),
                    name=meta.get("name", "tool"),
                    input=_parse_tool_input(args),
                ),
            )
            yield anthropic_models.ContentBlockStopEvent(
                type="content_block_stop", index=current_index
            )
            current_index += 1

        elif event_type == "response.output_item.added":
            item = event.get("item") or {}
            item_type = item.get("type")
            if item_type == "output_tool_call":
                output_index = item.get("output_index", 0)
                tool_meta[output_index] = {
                    "id": item.get("id", f"call_{output_index}"),
                    "name": item.get("name", "tool"),
                }
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
                output_index = item.get("output_index", 0)
                args = "".join(tool_args.pop(output_index, []))
                meta = tool_meta.pop(
                    output_index,
                    {
                        "id": item.get("id", f"call_{output_index}"),
                        "name": item.get("name", "tool"),
                    },
                )
                yield anthropic_models.ContentBlockStartEvent(
                    type="content_block_start",
                    index=current_index,
                    content_block=anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id=meta.get("id", f"call_{output_index}"),
                        name=meta.get("name", "tool"),
                        input=_parse_tool_input(args),
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

        async for chunk in stream:
            # Handle both dict and typed model inputs
            if isinstance(chunk, dict):
                if not chunk.get("choices"):
                    continue
                choices = chunk["choices"]
                if not choices:
                    continue
                choice = choices[0]
                model_id = chunk.get("model", model_id)
            else:
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                model_id = chunk.model or model_id

            # Start message if not started
            if not message_started:
                chunk_id = (
                    chunk.get("id", "msg_stream")
                    if isinstance(chunk, dict)
                    else (chunk.id or "msg_stream")
                )
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

            # Handle content delta and tool calls - support both dict and typed formats
            content = None
            finish_reason = None
            tool_calls = None

            if isinstance(chunk, dict):
                if choice.get("delta") and choice["delta"].get("content"):
                    content = choice["delta"]["content"]
                finish_reason = choice.get("finish_reason")
                tool_calls = choice.get("delta", {}).get("tool_calls") or choice.get(
                    "tool_calls"
                )
            else:
                if choice.delta and choice.delta.content:
                    content = choice.delta.content
                finish_reason = choice.finish_reason
                tool_calls = getattr(choice.delta, "tool_calls", None) or getattr(
                    choice, "tool_calls", None
                )

            if content:
                accumulated_content += content

                # Start content block if not started
                if not text_block_started:
                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=0,
                        content_block=anthropic_models.TextBlock(type="text", text=""),
                    )
                    text_block_started = True

                # Emit content delta
                yield anthropic_models.ContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=current_index,
                    delta=anthropic_models.TextBlock(type="text", text=content),
                )

            # Handle tool calls (strict JSON parsing)
            if tool_calls and isinstance(tool_calls, list):
                # Close any active text block before emitting tool_use
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=current_index
                    )
                    text_block_started = False
                    current_index += 1
                for i, tc in enumerate(tool_calls):
                    fn = None
                    if isinstance(tc, dict):
                        fn = tc.get("function")
                        tool_id = tc.get("id") or f"call_{i}"
                        name = fn.get("name") if isinstance(fn, dict) else None
                        args_raw = fn.get("arguments") if isinstance(fn, dict) else None
                    else:
                        fn = getattr(tc, "function", None)
                        tool_id = getattr(tc, "id", None) or f"call_{i}"
                        name = getattr(fn, "name", None) if fn is not None else None
                        args_raw = (
                            getattr(fn, "arguments", None) if fn is not None else None
                        )
                    from ccproxy.llms.formatters.shared.utils import (
                        strict_parse_tool_arguments,
                    )

                    args = strict_parse_tool_arguments(args_raw)
                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=current_index,
                        content_block=anthropic_models.ToolUseBlock(
                            type="tool_use",
                            id=tool_id,
                            name=name or "function",
                            input=args,
                        ),
                    )
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=current_index
                    )
                    current_index += 1

            # Handle finish reason
            if finish_reason:
                # Stop content block if started
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=current_index
                    )
                    text_block_started = False
                    current_index += 1

                # Map OpenAI finish reason to Anthropic stop reason via shared utility
                from ccproxy.llms.formatters.shared.utils import (
                    map_openai_finish_to_anthropic_stop,
                )

                stop_reason = map_openai_finish_to_anthropic_stop(finish_reason)

                # Get usage if available
                if isinstance(chunk, dict):
                    usage = chunk.get("usage")
                    anthropic_usage = (
                        anthropic_models.Usage(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                        )
                        if usage
                        else anthropic_models.Usage(input_tokens=0, output_tokens=0)
                    )
                else:
                    usage = getattr(chunk, "usage", None)
                    anthropic_usage = (
                        anthropic_models.Usage(
                            input_tokens=usage.prompt_tokens,
                            output_tokens=usage.completion_tokens,
                        )
                        if usage
                        else anthropic_models.Usage(input_tokens=0, output_tokens=0)
                    )

                # Emit message delta and stop
                yield anthropic_models.MessageDeltaEvent(
                    type="message_delta",
                    delta=anthropic_models.MessageDelta(
                        stop_reason=map_openai_finish_to_anthropic_stop(stop_reason)
                    ),
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
        stop_reason=map_openai_finish_to_anthropic_stop(stop_reason),
        stop_sequence=None,
        usage=usage,
    )
