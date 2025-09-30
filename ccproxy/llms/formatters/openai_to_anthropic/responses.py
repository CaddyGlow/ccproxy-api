"""Response conversion entry points for OpenAI→Anthropic adapters."""

from __future__ import annotations

from typing import cast

import ccproxy.core.logging
from ccproxy.llms.formatters.common import (
    THINKING_PATTERN,
    convert_openai_responses_usage_to_anthropic_usage,
    convert_openai_responses_usage_to_completion_usage,
)
from ccproxy.llms.formatters.utils import (
    map_openai_finish_to_anthropic_stop,
    openai_usage_to_anthropic_usage,
    strict_parse_tool_arguments,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models


logger = ccproxy.core.logging.get_logger(__name__)


def convert__openai_responses_usage_to_openai_completion__usage(
    usage: openai_models.ResponseUsage,
) -> openai_models.CompletionUsage:
    return convert_openai_responses_usage_to_completion_usage(usage)


def convert__openai_responses_usage_to_anthropic__usage(
    usage: openai_models.ResponseUsage,
) -> anthropic_models.Usage:
    return convert_openai_responses_usage_to_anthropic_usage(usage)


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


__all__ = [
    "convert__openai_chat_to_anthropic_messages__response",
    "convert__openai_responses_to_anthropic_message__response",
    "convert__openai_responses_usage_to_anthropic__usage",
    "convert__openai_responses_usage_to_openai_completion__usage",
]
