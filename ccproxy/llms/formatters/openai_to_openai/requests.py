"""Request conversion entry points for OpenAI↔OpenAI adapters."""

from __future__ import annotations

import contextlib
import json
import os
from typing import Any

import ccproxy.core.logging
from ccproxy.llms.formatters.context import (
    register_request,
    register_request_tools,
)
from ccproxy.llms.models import openai as openai_models

from ._helpers import (
    _collect_chat_instruction_segments,
    _convert_tool_choice_chat_to_responses,
    _convert_tool_choice_responses_to_chat,
    _convert_tools_chat_to_responses,
    _convert_tools_responses_to_chat,
    _extract_responses_role_and_content,
    _get_attr,
    _normalize_responses_input_content,
)


logger = ccproxy.core.logging.get_logger(__name__)


async def convert__openai_responses_to_openaichat__request(
    request: openai_models.ResponseRequest,
) -> openai_models.ChatCompletionRequest:
    """Convert a Responses API request into a ChatCompletionRequest."""

    _log = logger.bind(category="formatter", converter="responses_to_chat_request")
    system_segments: list[str] = []
    messages: list[dict[str, Any]] = []
    tool_call_aliases: dict[str, str] = {}
    fallback_tool_index = 0

    if isinstance(request.instructions, str) and request.instructions.strip():
        system_segments.append(request.instructions.strip())

    if isinstance(request.input, str):
        user_text = request.input.strip()
        if user_text:
            messages.append({"role": "user", "content": user_text})
    else:
        for item in request.input or []:
            item_type_raw = _get_attr(item, "type")
            item_type = (
                item_type_raw.lower() if isinstance(item_type_raw, str) else None
            )

            if item_type in {"function_call", "tool_call"}:
                call_identifier = _get_attr(item, "call_id") or _get_attr(item, "id")
                if not isinstance(call_identifier, str) or not call_identifier:
                    call_identifier = f"call_{fallback_tool_index}"
                    fallback_tool_index += 1

                tool_call_aliases[str(call_identifier)] = str(call_identifier)

                function_block = _get_attr(item, "function")
                name = (
                    _get_attr(function_block, "name") or _get_attr(item, "name") or ""
                )
                arguments_value = _get_attr(function_block, "arguments")
                if arguments_value is None:
                    arguments_value = _get_attr(item, "arguments")

                arguments_text: str
                if isinstance(arguments_value, str) and arguments_value.strip():
                    arguments_text = arguments_value
                elif isinstance(arguments_value, dict | list):
                    arguments_text = json.dumps(arguments_value, ensure_ascii=False)
                elif arguments_value is None:
                    arguments_text = "{}"
                else:
                    arguments_text = ""
                    with contextlib.suppress(TypeError, ValueError):
                        arguments_text = json.dumps(arguments_value, ensure_ascii=False)
                    if (
                        not isinstance(arguments_text, str)
                        or not arguments_text.strip()
                    ):
                        arguments_text = str(arguments_value)
                    if (
                        not isinstance(arguments_text, str)
                        or not arguments_text.strip()
                    ):
                        arguments_text = "{}"

                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": str(call_identifier),
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": arguments_text,
                                },
                            }
                        ],
                    }
                )
                continue

            if item_type in {"function_call_output", "tool_output", "tool_response"}:
                call_identifier = _get_attr(item, "call_id") or _get_attr(item, "id")
                mapped_identifier = tool_call_aliases.get(str(call_identifier))
                if mapped_identifier is None:
                    mapped_identifier = str(
                        call_identifier or f"call_{fallback_tool_index}"
                    )
                    if mapped_identifier.startswith("call_"):
                        fallback_tool_index += 1
                    tool_call_aliases[str(call_identifier or mapped_identifier)] = (
                        mapped_identifier
                    )

                output_value = _get_attr(item, "output")
                if output_value is None:
                    output_value = _get_attr(item, "content")
                if output_value is None:
                    output_value = _get_attr(item, "text")

                output_text = _normalize_responses_input_content(output_value)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(mapped_identifier),
                        "content": output_text or "",
                    }
                )
                continue

            role, raw_content = _extract_responses_role_and_content(item)
            normalized_role = role.lower() if isinstance(role, str) else None

            if normalized_role == "system":
                if raw_content:
                    content_text = _normalize_responses_input_content(raw_content)
                    if content_text:
                        system_segments.append(content_text)
                continue

            content_text = _normalize_responses_input_content(raw_content)

            if normalized_role in {"assistant", "tool"}:
                messages.append(
                    {
                        "role": normalized_role,
                        "content": content_text,
                    }
                )
                continue

            if normalized_role == "developer":
                if content_text:
                    system_segments.append(content_text)
                continue

            if normalized_role not in {"user"}:
                normalized_role = "user"

            messages.append({"role": normalized_role, "content": content_text})

    if system_segments:
        merged_system = "\n\n".join(
            segment for segment in system_segments if segment
        ).strip()
        if merged_system:
            messages.insert(0, {"role": "system", "content": merged_system})

    if not messages:
        messages.append({"role": "user", "content": "(empty request)"})

    payload: dict[str, Any] = {
        "model": request.model or "gpt-4o-mini",
        "messages": messages,
    }

    reasoning_cfg = getattr(request, "reasoning", None)
    effort_value: Any = None
    if isinstance(reasoning_cfg, dict):
        effort_value = reasoning_cfg.get("effort")
    elif reasoning_cfg is not None:
        effort_value = _get_attr(reasoning_cfg, "effort")
    if isinstance(effort_value, str) and effort_value:
        payload["reasoning_effort"] = effort_value

    with contextlib.suppress(Exception):
        _log.debug(
            "responses_to_chat_compiled_messages",
            message_count=len(messages),
            roles=[m.get("role") for m in messages],
        )

    if request.max_output_tokens is not None:
        payload["max_completion_tokens"] = request.max_output_tokens

    if request.stream is not None:
        payload["stream"] = request.stream

    if request.temperature is not None:
        payload["temperature"] = request.temperature

    if request.top_p is not None:
        payload["top_p"] = request.top_p

    tools = _convert_tools_responses_to_chat(request.tools)
    if tools:
        payload["tools"] = tools

    if request.tool_choice is not None:
        payload["tool_choice"] = _convert_tool_choice_responses_to_chat(
            request.tool_choice
        )

    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls

    return openai_models.ChatCompletionRequest.model_validate(payload)


def _build_responses_payload_from_chat_request(
    request: openai_models.ChatCompletionRequest,
) -> tuple[dict[str, Any], str | None]:
    """Project a ChatCompletionRequest into Responses payload fields."""

    payload_data: dict[str, Any] = {"model": request.model}
    instructions_text: str | None = None

    if request.max_completion_tokens is not None:
        payload_data["max_output_tokens"] = int(request.max_completion_tokens)

    user_text: str | None = None
    for msg in reversed(request.messages):
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            user_text = msg.content
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
                user_text = " ".join(texts)
        break

    if user_text:
        payload_data["input"] = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_text},
                ],
            }
        ]
    else:
        payload_data["input"] = []

    instruction_segments = _collect_chat_instruction_segments(request.messages)
    instructions_text = "\n\n".join(
        segment for segment in instruction_segments if segment
    )
    if instructions_text:
        payload_data["instructions"] = instructions_text

    resp_fmt = request.response_format
    if resp_fmt is not None:
        if resp_fmt.type == "text":
            payload_data["text"] = {"format": {"type": "text"}}
        elif resp_fmt.type == "json_object":
            payload_data["text"] = {"format": {"type": "json_object"}}
        elif resp_fmt.type == "json_schema" and hasattr(resp_fmt, "json_schema"):
            js = resp_fmt.json_schema
            fmt = {"type": "json_schema"}
            if js is not None:
                js_dict = js.model_dump() if hasattr(js, "model_dump") else js
                if isinstance(js_dict, dict):
                    fmt.update(
                        {
                            key: value
                            for key, value in js_dict.items()
                            if key
                            in {"name", "schema", "strict", "$defs", "description"}
                        }
                    )
            payload_data["text"] = {"format": fmt}

    tools = _convert_tools_chat_to_responses(request.tools)
    if tools:
        payload_data["tools"] = tools

    if request.tool_choice is not None:
        payload_data["tool_choice"] = _convert_tool_choice_chat_to_responses(
            request.tool_choice
        )

    if request.parallel_tool_calls is not None:
        payload_data["parallel_tool_calls"] = bool(request.parallel_tool_calls)

    if request.temperature is not None:
        payload_data["temperature"] = request.temperature

    if request.top_p is not None:
        payload_data["top_p"] = request.top_p

    if request.top_logprobs is not None:
        payload_data["top_logprobs"] = request.top_logprobs

    if request.service_tier is not None:
        payload_data["service_tier"] = request.service_tier

    if request.store is not None:
        payload_data["store"] = request.store

    if request.prompt_cache_key:
        payload_data["prompt_cache_key"] = request.prompt_cache_key

    if request.user:
        payload_data["user"] = request.user

    reasoning_effort = None
    if isinstance(request.reasoning_effort, str) and request.reasoning_effort:
        reasoning_effort = request.reasoning_effort
    else:
        env_toggle = os.getenv("LLM__OPENAI_THINKING_XML")
        if env_toggle is None:
            env_toggle = os.getenv("OPENAI_STREAM_ENABLE_THINKING_SERIALIZATION")
        enable_thinking = True
        if env_toggle is not None:
            enable_thinking = env_toggle.strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
        if enable_thinking:
            reasoning_effort = "medium"

    if reasoning_effort:
        payload_data["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}

    payload_data.setdefault("background", None)

    return payload_data, instructions_text or None


async def convert__openai_chat_to_openai_responses__request(
    request: openai_models.ChatCompletionRequest,
) -> openai_models.ResponseRequest:
    """Convert ChatCompletionRequest to ResponseRequest using typed models."""

    payload_data, instructions_text = _build_responses_payload_from_chat_request(
        request
    )

    response_request = openai_models.ResponseRequest.model_validate(payload_data)

    register_request_tools(request.tools)
    register_request(request, instructions_text)

    return response_request


__all__ = [
    "convert__openai_chat_to_openai_responses__request",
    "convert__openai_responses_to_openaichat__request",
]
