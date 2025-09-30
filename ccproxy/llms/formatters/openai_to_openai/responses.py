"""Response conversion entry points for OpenAI↔OpenAI adapters."""

from __future__ import annotations

import contextlib
import json
import time
from typing import Any, Literal

import ccproxy.core.logging
from ccproxy.llms.formatters.common import (
    THINKING_PATTERN,
    ThinkingSegment,
    convert_openai_completion_usage_to_responses_usage,
    convert_openai_responses_usage_to_completion_usage,
    merge_thinking_segments,
)
from ccproxy.llms.models import openai as openai_models

from ._helpers import (
    _get_attr,
)


logger = ccproxy.core.logging.get_logger(__name__)


def convert__openai_responses_usage_to_openai_completion__usage(
    usage: openai_models.ResponseUsage,
) -> openai_models.CompletionUsage:
    return convert_openai_responses_usage_to_completion_usage(usage)


def convert__openai_completion_usage_to_openai_responses__usage(
    usage: openai_models.CompletionUsage,
) -> openai_models.ResponseUsage:
    return convert_openai_completion_usage_to_responses_usage(usage)


def _adopt_summary_entry(entry: Any) -> dict[str, Any] | None:
    """Conversion of arbitrary summary nodes to dicts."""

    if isinstance(entry, dict):
        return entry

    if hasattr(entry, "model_dump"):
        with contextlib.suppress(Exception):
            data = entry.model_dump(mode="json", exclude_none=True)
            if isinstance(data, dict):
                return data
        with contextlib.suppress(Exception):
            data = entry.model_dump()
            if isinstance(data, dict):
                return data

    if hasattr(entry, "__dict__"):
        dict_data: dict[str, Any] = {}
        for key in (
            "type",
            "text",
            "content",
            "signature",
            "summary",
            "value",
            "delta",
            "reasoning",
        ):
            if hasattr(entry, key):
                value = getattr(entry, key)
                if value is not None:
                    dict_data[key] = value
        if dict_data:
            return dict_data
    return None


def _collect_reasoning_segments(source: Any) -> list[ThinkingSegment]:
    if source is None:
        return []

    segments: list[ThinkingSegment] = []
    visited: set[int] = set()

    def _walk(node: Any, inherited_signature: str | None) -> None:
        if node is None:
            return

        # if isinstance(node, str):
        #     if node:
        #         normalized = node.strip().lower()
        #         if normalized and normalized not in _REASONING_SUMMARY_MODES:
        #             segments.append(
        #                 ThinkingSegment(thinking=node, signature=inherited_signature)
        #             )
        #     return

        if isinstance(node, bytes | bytearray):
            try:
                decoded = node.decode()
            except UnicodeDecodeError:
                return
            if decoded:
                segments.append(
                    ThinkingSegment(thinking=decoded, signature=inherited_signature)
                )
            return

        if isinstance(node, list | tuple | set):
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            start_idx = len(segments)
            current_signature = inherited_signature
            for child in node:
                child_data = _adopt_summary_entry(child)
                child_type = (
                    child_data.get("type")
                    if isinstance(child_data, dict)
                    else _get_attr(child, "type")
                )
                if child_type == "signature":
                    candidate = None
                    if isinstance(child_data, dict):
                        candidate = child_data.get("text") or child_data.get(
                            "signature"
                        )
                    else:
                        candidate = _get_attr(child, "text") or _get_attr(
                            child, "signature"
                        )
                    if isinstance(candidate, str) and candidate:
                        current_signature = candidate
                        for idx in range(start_idx, len(segments)):
                            segments[idx] = ThinkingSegment(
                                thinking=segments[idx].thinking,
                                signature=current_signature,
                            )
                        start_idx = len(segments)
                _walk(child, current_signature)
            return

        if isinstance(node, dict) or hasattr(node, "__dict__"):
            current_node_id = id(node)
            if current_node_id in visited:
                return
            visited.add(current_node_id)

        data = _adopt_summary_entry(node)
        if data is None:
            text_attr = _get_attr(node, "text")
            signature_attr = _get_attr(node, "signature")
            type_attr = _get_attr(node, "type")
            next_signature = inherited_signature
            if isinstance(signature_attr, str) and signature_attr:
                next_signature = signature_attr
            if type_attr == "signature" and isinstance(text_attr, str) and text_attr:
                next_signature = text_attr
                text_attr = None
            if isinstance(text_attr, str) and text_attr:
                segments.append(
                    ThinkingSegment(thinking=text_attr, signature=next_signature)
                )

            for key in ("summary", "content"):
                nested = _get_attr(node, key)
                if isinstance(nested, list | tuple | set):
                    for child in nested:
                        _walk(child, next_signature)
                elif isinstance(nested, dict):
                    _walk(nested, next_signature)
            return

        node_type = data.get("type")
        text_value = data.get("text")
        signature_value = data.get("signature")
        content_value = data.get("content")
        summary_value = data.get("summary")
        reasoning_value = data.get("reasoning")

        next_signature = inherited_signature
        if isinstance(signature_value, str) and signature_value:
            next_signature = signature_value

        if node_type == "signature":
            if isinstance(text_value, str) and text_value:
                next_signature = text_value
            if isinstance(content_value, list | tuple | set):
                for child in content_value:
                    _walk(child, next_signature)
            return

        if node_type in {"summary_group", "group"}:
            if isinstance(content_value, list | tuple | set):
                start_idx = len(segments)
                current_signature = next_signature
                for child in content_value:
                    child_data = _adopt_summary_entry(child)
                    child_type = (
                        child_data.get("type")
                        if isinstance(child_data, dict)
                        else _get_attr(child, "type")
                    )
                    if child_type == "signature":
                        candidate = None
                        if isinstance(child_data, dict):
                            candidate = child_data.get("text") or child_data.get(
                                "signature"
                            )
                        else:
                            candidate = _get_attr(child, "text") or _get_attr(
                                child, "signature"
                            )
                        if isinstance(candidate, str) and candidate:
                            current_signature = candidate
                            for idx in range(start_idx, len(segments)):
                                segments[idx] = ThinkingSegment(
                                    thinking=segments[idx].thinking,
                                    signature=current_signature,
                                )
                            start_idx = len(segments)
                    _walk(child, current_signature)
            return

        emitted = False
        if node_type in {"summary_text", "text", "reasoning_text"}:
            if isinstance(text_value, str) and text_value:
                segments.append(
                    ThinkingSegment(thinking=text_value, signature=next_signature)
                )
                emitted = True
        elif (
            isinstance(text_value, str)
            and text_value
            and node_type not in {"signature"}
        ):
            segments.append(
                ThinkingSegment(thinking=text_value, signature=next_signature)
            )
            emitted = True

        value_value = data.get("value")
        if not emitted and isinstance(value_value, str) and value_value:
            segments.append(
                ThinkingSegment(thinking=value_value, signature=next_signature)
            )
            emitted = True

        if isinstance(summary_value, list | tuple | set):
            for child in summary_value:
                _walk(child, next_signature)
        elif isinstance(summary_value, dict):
            _walk(summary_value, next_signature)

        if isinstance(content_value, list | tuple | set):
            for child in content_value:
                _walk(child, next_signature)
        elif isinstance(content_value, dict):
            _walk(content_value, next_signature)

        if isinstance(reasoning_value, list | tuple | set | dict):
            _walk(reasoning_value, next_signature)

    _walk(source, None)
    return merge_thinking_segments(segments)


def _wrap_thinking(signature: str | None, text: str) -> str:
    """Serialize a reasoning block into <thinking> XML."""
    return ThinkingSegment(thinking=text, signature=signature).to_xml()


def _extract_reasoning_blocks(payload: Any) -> list[ThinkingSegment]:
    """Extract reasoning blocks from a response output payload."""

    if not payload:
        return []

    summary = _get_attr(payload, "summary")
    segments = _collect_reasoning_segments(summary)
    if segments:
        return segments

    if isinstance(payload, list | tuple | set):
        segments = _collect_reasoning_segments(payload)
        if segments:
            return segments

    text_value = _get_attr(payload, "text")
    if isinstance(text_value, str) and text_value:
        return [ThinkingSegment(thinking=text_value)]

    if isinstance(payload, dict):
        raw = payload.get("reasoning")
        if raw:
            return _extract_reasoning_blocks(raw)

    return []


def _split_content_segments(content: str) -> list[tuple[str, Any]]:
    """Split mixed assistant content into ordered text vs thinking blocks."""

    if not content:
        return []

    segments: list[tuple[str, Any]] = []
    last_idx = 0
    for match in THINKING_PATTERN.finditer(content):
        start, end = match.span()
        if start > last_idx:
            text_segment = content[last_idx:start]
            if text_segment:
                segments.append(("text", text_segment))
        signature = match.group(1) or None
        thinking_text = match.group(2) or ""
        segments.append(
            ("thinking", ThinkingSegment.from_xml(signature, thinking_text))
        )
        last_idx = end

    if last_idx < len(content):
        tail = content[last_idx:]
        if tail:
            segments.append(("text", tail))

    if not segments:
        segments.append(("text", content))
    return segments


def convert__openai_responses_to_openai_chat__response(
    response: openai_models.ResponseObject,
) -> openai_models.ChatCompletionResponse:
    """Convert an OpenAI ResponseObject to a ChatCompletionResponse."""
    text_segments: list[str] = []
    added_reasoning: set[tuple[str, str]] = set()
    tool_calls: list[openai_models.ToolCall] = []

    for item in response.output or []:
        logger.debug(
            "convert_responses_to_chat_response_item", item_type=_get_attr(item, "type")
        )
        item_type = _get_attr(item, "type")
        if item_type == "reasoning":
            for segment in _extract_reasoning_blocks(item):
                signature = segment.signature
                thinking_text = segment.thinking
                logger.debug(
                    "convert_responses_to_chat_reasoning_block",
                    signature=signature,
                    text_snippet=(thinking_text[:30] + "...")
                    if thinking_text and len(thinking_text) > 30
                    else thinking_text,
                )
                if thinking_text:
                    key = (signature or "", thinking_text)
                    if key not in added_reasoning:
                        text_segments.append(_wrap_thinking(signature, thinking_text))
                        added_reasoning.add(key)
        elif item_type == "message":
            parts: list[str] = []
            content_list = _get_attr(item, "content")
            if isinstance(content_list, list):
                for part in content_list:
                    part_type = _get_attr(part, "type")
                    if part_type == "output_text":
                        text_val = _get_attr(part, "text")
                        if isinstance(text_val, str):
                            parts.append(text_val)
                    elif isinstance(part, str):
                        parts.append(part)
            elif isinstance(content_list, str):
                parts.append(content_list)
            if parts:
                text_segments.append("".join(parts))
        elif item_type == "function_call":
            function_block = _get_attr(item, "function")
            name = _get_attr(function_block, "name") or _get_attr(item, "name")
            arguments_value: Any = _get_attr(item, "arguments")
            if arguments_value is None and isinstance(function_block, dict):
                arguments_value = function_block.get("arguments")

            if not isinstance(name, str) or not name:
                continue

            if isinstance(arguments_value, dict):
                arguments_str = json.dumps(arguments_value)
            elif isinstance(arguments_value, str):
                arguments_str = arguments_value
            else:
                arguments_str = json.dumps(arguments_value or {})

            tool_calls.append(
                openai_models.ToolCall(
                    id=_get_attr(item, "id")
                    or _get_attr(item, "call_id")
                    or f"call_{len(tool_calls)}",
                    type="function",
                    function=openai_models.FunctionCall(
                        name=name,
                        arguments=arguments_str,
                    ),
                )
            )

    text_content = "".join(text_segments)

    usage = None
    if response.usage:
        usage = convert__openai_responses_usage_to_openai_completion__usage(
            response.usage
        )

    finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = (
        "tool_calls" if tool_calls else "stop"
    )

    return openai_models.ChatCompletionResponse(
        id=response.id or "chatcmpl-resp",
        choices=[
            openai_models.Choice(
                index=0,
                message=openai_models.ResponseMessage(
                    role="assistant",
                    content=text_content,
                    tool_calls=tool_calls or None,
                ),
                finish_reason=finish_reason,
            )
        ],
        created=0,
        model=response.model or "",
        object="chat.completion",
        usage=usage
        or openai_models.CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        ),
    )


async def convert__openai_chat_to_openai_responses__response(
    chat_response: openai_models.ChatCompletionResponse,
) -> openai_models.ResponseObject:
    content_text = ""
    tool_calls: list[Any] = []
    if chat_response.choices:
        first_choice = chat_response.choices[0]
        if first_choice.message:
            content = first_choice.message.content
            if content:
                if isinstance(content, str):
                    content_text = content
                elif isinstance(content, list):
                    # Handle list content - convert to string
                    content_text = str(content)
                else:
                    content_text = str(content)
            if first_choice.message.tool_calls:
                tool_calls = list(first_choice.message.tool_calls)

    segments = _split_content_segments(content_text)

    outputs: list[Any] = []
    reasoning_entries: list[Any] = []
    message_buffer: list[str] = []
    message_counter = 0

    def flush_message() -> None:
        nonlocal message_buffer, message_counter
        if not message_buffer:
            return
        message_text = "".join(message_buffer)
        message_buffer = []
        message_id = f"msg_{chat_response.id or 'unknown'}_{message_counter}"
        message_counter += 1
        outputs.append(
            openai_models.MessageOutput(
                type="message",
                role="assistant",
                id=message_id,
                status="completed",
                content=[
                    openai_models.OutputTextContent(
                        type="output_text", text=message_text
                    )
                ],
            )
        )

    for segment in segments or [("text", "")]:
        if not segment:
            continue
        kind = segment[0]
        if kind == "text":
            text_part = segment[1]
            if isinstance(text_part, str) and text_part:
                message_buffer.append(text_part)
        elif kind == "thinking":
            segment_value = segment[1]
            if isinstance(segment_value, ThinkingSegment):
                signature = segment_value.signature
                thinking_text = segment_value.thinking
            else:
                signature = None
                thinking_text = ""
            flush_message()
            summary_entry: dict[str, Any] = {
                "type": "summary_text",
                "text": thinking_text,
            }
            if signature:
                summary_entry["signature"] = signature
            reasoning_id = (
                f"reasoning_{chat_response.id or 'unknown'}_{len(reasoning_entries)}"
            )
            reasoning_output = openai_models.ReasoningOutput(
                type="reasoning",
                id=reasoning_id,
                status="completed",
                summary=[summary_entry],
            )
            outputs.append(reasoning_output)
            reasoning_entries.append(reasoning_output)

    # Flush any remaining assistant text
    flush_message()

    if not outputs:
        outputs.append(
            openai_models.MessageOutput(
                type="message",
                role="assistant",
                id=f"msg_{chat_response.id or 'unknown'}_0",
                status="completed",
                content=[openai_models.OutputTextContent(type="output_text", text="")],
            )
        )

    if tool_calls:
        for idx, tool_call in enumerate(tool_calls):
            fn = getattr(tool_call, "function", None)
            name = _get_attr(fn, "name") or _get_attr(tool_call, "name") or ""
            arguments = _get_attr(fn, "arguments") or _get_attr(tool_call, "arguments")
            if isinstance(arguments, dict):
                arguments_value: str | dict[str, Any] | None = arguments
            else:
                arguments_value = str(arguments) if arguments is not None else None
            outputs.append(
                openai_models.FunctionCallOutput(
                    type="function_call",
                    id=getattr(tool_call, "id", f"call_{idx}"),
                    status="completed",
                    name=name,
                    call_id=getattr(tool_call, "id", None),
                    arguments=arguments_value,
                )
            )

    reasoning_summary = []
    for entry in reasoning_entries:
        summary_list = _get_attr(entry, "summary")
        if isinstance(summary_list, list):
            reasoning_summary.extend(summary_list)

    usage: openai_models.ResponseUsage | None = None
    if chat_response.usage:
        usage = convert__openai_completion_usage_to_openai_responses__usage(
            chat_response.usage
        )

    return openai_models.ResponseObject(
        id=chat_response.id or "resp-unknown",
        object="response",
        created_at=int(time.time()),
        model=chat_response.model or "",
        status="completed",
        output=outputs,
        parallel_tool_calls=False,
        usage=usage,
        reasoning=(
            openai_models.Reasoning(summary=reasoning_summary)
            if reasoning_summary
            else None
        ),
    )


__all__ = [
    "convert__openai_chat_to_openai_responses__response",
    "convert__openai_completion_usage_to_openai_responses__usage",
    "convert__openai_responses_to_openai_chat__response",
    "convert__openai_responses_usage_to_openai_completion__usage",
]
