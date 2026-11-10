"""Response conversion entry points for Anthropic→OpenAI adapters."""

from __future__ import annotations

import time
from typing import Any

import ccproxy.core.logging
from ccproxy.llms.formatters.common import (
    convert_anthropic_usage_to_openai_completion_usage,
    convert_anthropic_usage_to_openai_responses_usage,
)
from ccproxy.llms.formatters.constants import ANTHROPIC_TO_OPENAI_FINISH_REASON
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models

from ._helpers import build_openai_tool_call


logger = ccproxy.core.logging.get_logger(__name__)


def convert__anthropic_usage_to_openai_completion__usage(
    usage: anthropic_models.Usage,
) -> openai_models.CompletionUsage:
    return convert_anthropic_usage_to_openai_completion_usage(usage)


def convert__anthropic_usage_to_openai_responses__usage(
    usage: anthropic_models.Usage,
) -> openai_models.ResponseUsage:
    return convert_anthropic_usage_to_openai_responses_usage(usage)


def convert__anthropic_message_to_openai_responses__response(
    response: anthropic_models.MessageResponse,
) -> openai_models.ResponseObject:
    """Convert Anthropic MessageResponse to an OpenAI ResponseObject."""
    text_parts: list[str] = []
    tool_contents: list[dict[str, Any]] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            thinking = getattr(block, "thinking", None) or ""
            text_parts.append(f"<think>{thinking}</think>")
        elif block_type == "redacted_thinking":
            # Skip redacted thinking blocks
            continue
        elif block_type == "tool_use":
            tool_contents.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", "tool_1"),
                    "name": getattr(block, "name", "function"),
                    "arguments": getattr(block, "input", {}) or {},
                }
            )

    message_content: list[dict[str, Any]] = []
    if text_parts:
        message_content.append(
            openai_models.OutputTextContent(
                type="output_text",
                text="".join(text_parts),
            ).model_dump()
        )
    message_content.extend(tool_contents)

    usage_model = None
    if response.usage is not None:
        usage_model = convert__anthropic_usage_to_openai_responses__usage(
            response.usage
        )

    return openai_models.ResponseObject(
        id=response.id,
        object="response",
        created_at=0,
        status="completed",
        model=response.model,
        output=[
            openai_models.MessageOutput(
                type="message",
                id=f"{response.id}_msg_0",
                status="completed",
                role="assistant",
                content=message_content,  # type: ignore[arg-type]
            )
        ],
        parallel_tool_calls=False,
        usage=usage_model,
    )


def convert__anthropic_message_to_openai_chat__response(
    response: anthropic_models.MessageResponse,
) -> openai_models.ChatCompletionResponse:
    """Convert Anthropic MessageResponse to an OpenAI ChatCompletionResponse."""
    content_blocks = response.content
    parts: list[str] = []
    tool_calls: list[openai_models.ToolCall] = []

    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        elif btype == "thinking":
            thinking = getattr(block, "thinking", None)
            if isinstance(thinking, str):
                parts.append(f"<think>{thinking}</think>")
        # Skip redacted_thinking blocks
        elif btype == "redacted_thinking":
            continue
        elif btype == "tool_use":
            tool_calls.append(
                build_openai_tool_call(
                    tool_id=getattr(block, "id", None),
                    tool_name=getattr(block, "name", None),
                    tool_input=getattr(block, "input", {}) or {},
                    fallback_index=len(tool_calls),
                )
            )

    content_text = "".join(parts) if parts else None

    stop_reason = response.stop_reason
    finish_reason = ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
        stop_reason or "end_turn", "stop"
    )

    usage_model = convert__anthropic_usage_to_openai_completion__usage(response.usage)

    message_dict: dict[str, Any] = {"role": "assistant", "content": content_text}
    if tool_calls:
        message_dict["tool_calls"] = [call.model_dump() for call in tool_calls]

    payload = {
        "id": response.id,
        "choices": [
            {
                "index": 0,
                "message": message_dict,
                "finish_reason": finish_reason,
            }
        ],
        "created": int(time.time()),
        "model": response.model,
        "object": "chat.completion",
        "usage": usage_model.model_dump(),
    }

    return openai_models.ChatCompletionResponse.model_validate(payload)


__all__ = [
    "convert__anthropic_message_to_openai_chat__response",
    "convert__anthropic_message_to_openai_responses__response",
    "convert__anthropic_usage_to_openai_completion__usage",
    "convert__anthropic_usage_to_openai_responses__usage",
]
