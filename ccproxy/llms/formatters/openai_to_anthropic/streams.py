"""OpenAI→Anthropic streaming conversion entry points."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, cast

import ccproxy.core.logging
from ccproxy.llms.formatters.common import (
    IndexedToolCallTracker,
    ToolCallTracker,
    emit_anthropic_tool_use_events,
)
from ccproxy.llms.formatters.utils import (
    map_openai_finish_to_anthropic_stop,
    openai_usage_to_anthropic_usage,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import OpenAIAccumulator


logger = ccproxy.core.logging.get_logger(__name__)


class OpenAIResponsesToAnthropicStreamAdapter:
    """Stateful adapter for OpenAI Responses → Anthropic streaming."""

    async def run(
        self,
        stream: AsyncIterator[Any],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        async for event in self._convert_responses_stream(stream):
            yield event

    async def _convert_responses_stream(
        self,
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
        tool_states = IndexedToolCallTracker()

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
                            content_block=anthropic_models.TextBlock(
                                type="text", text=""
                            ),
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
                output_index = int(event.get("output_index", 0) or 0)
                state = tool_states.ensure(output_index)
                if state.output_index < 0:
                    state.output_index = output_index

                delta = event.get("delta") or {}
                delta_text = (
                    delta.get("arguments") if isinstance(delta, dict) else delta
                )
                if isinstance(delta_text, str):
                    state.add_arguments_part(delta_text)
                    state.append_arguments(delta_text)

                item_id_val = event.get("item_id")
                if isinstance(item_id_val, str) and item_id_val:
                    if not state.item_id:
                        state.item_id = item_id_val

                name_val = event.get("name")
                if isinstance(name_val, str) and name_val:
                    if not state.name:
                        state.name = name_val

                call_id_val = event.get("call_id")
                if isinstance(call_id_val, str) and call_id_val:
                    if not state.call_id:
                        state.call_id = call_id_val

            elif event_type == "response.function_call_arguments.done":
                output_index = int(event.get("output_index", 0) or 0)
                state = tool_states.ensure(output_index)
                if state.output_index < 0:
                    state.output_index = output_index

                arguments = event.get("arguments") if isinstance(event, dict) else None
                if isinstance(arguments, str) and arguments:
                    state.append_arguments(arguments)
                    state.final_arguments = arguments
                elif not state.final_arguments:
                    combined = state.arguments or "".join(state.arguments_parts)
                    if combined:
                        state.final_arguments = combined

                item_id_val = event.get("item_id")
                if isinstance(item_id_val, str) and item_id_val:
                    state.item_id = state.item_id or item_id_val

                name_val = event.get("name")
                if isinstance(name_val, str) and name_val:
                    state.name = state.name or name_val

                call_id_val = event.get("call_id")
                if isinstance(call_id_val, str) and call_id_val:
                    state.call_id = state.call_id or call_id_val

            elif event_type == "response.output_item.added":
                item = event.get("item") or {}
                item_type = item.get("type")
                if item_type == "output_tool_call":
                    output_index = int(
                        item.get("output_index", event.get("output_index", 0)) or 0
                    )
                    state = tool_states.ensure(output_index)
                    if state.output_index < 0:
                        state.output_index = output_index
                    item_id_val = item.get("id")
                    if isinstance(item_id_val, str) and item_id_val:
                        state.item_id = state.item_id or item_id_val
                    name_val = item.get("name")
                    if isinstance(name_val, str) and name_val:
                        state.name = state.name or name_val
                    call_id_val = item.get("call_id")
                    if isinstance(call_id_val, str) and call_id_val:
                        state.call_id = state.call_id or call_id_val

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
                    output_index = int(
                        item.get("output_index", event.get("output_index", 0)) or 0
                    )
                    state = tool_states.ensure(output_index)
                    if state.output_index < 0:
                        state.output_index = output_index

                    item_name = item.get("name")
                    if isinstance(item_name, str) and item_name:
                        state.name = state.name or item_name
                    item_call_id = item.get("call_id")
                    if isinstance(item_call_id, str) and item_call_id:
                        state.call_id = state.call_id or item_call_id
                    if item.get("id") and not state.item_id:
                        state.item_id = item.get("id")

                    args_str = (
                        state.final_arguments
                        or state.arguments
                        or "".join(state.arguments_parts)
                    )

                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=current_index,
                        content_block=anthropic_models.ToolUseBlock(
                            type="tool_use",
                            id=state.item_id or f"call_{state.index}",
                            name=state.name or "tool",
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


class OpenAIChatToAnthropicStreamAdapter:
    """Stateful adapter for OpenAI Chat → Anthropic streaming."""

    def run(
        self,
        stream: AsyncIterator[Any],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        return self._convert_chat_stream(stream)

    def _convert_chat_stream(
        self,
        stream: AsyncIterator[openai_models.ChatCompletionChunk],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        """Convert OpenAI ChatCompletion stream to Anthropic MessageStreamEvent stream."""

        async def generator() -> AsyncGenerator[
            anthropic_models.MessageStreamEvent, None
        ]:
            message_started = False
            text_block_started = False
            accumulated_content = ""
            model_id = ""
            current_index = 0
            tool_tracker = ToolCallTracker()
            openai_accumulator = OpenAIAccumulator()

            def _parse_tool_input(text: str) -> dict[str, Any]:
                if not text:
                    return {}
                try:
                    parsed = json.loads(text)
                    return parsed if isinstance(parsed, dict) else {"arguments": text}
                except Exception:
                    return {"arguments": text}

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
                            usage=anthropic_models.Usage(
                                input_tokens=0, output_tokens=0
                            ),
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
                            content_block=anthropic_models.TextBlock(
                                type="text", text=""
                            ),
                        )
                        text_block_started = True

                    yield anthropic_models.ContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=current_index,
                        delta=anthropic_models.TextBlock(
                            type="text", text=content_text
                        ),
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

                    complete_calls = openai_accumulator.get_complete_tool_calls()
                    for tool in complete_calls:
                        tool_id_value = tool.get("id")
                        call_identifier = str(
                            tool_id_value
                            or f"call_{tool.get('index', len(tool_tracker.values()))}"
                        )
                        state = tool_tracker.ensure(call_identifier)
                        if state.name is None:
                            state.name = tool.get("function", {}).get(
                                "name"
                            ) or tool.get("name")
                        if state.call_id is None and isinstance(tool_id_value, str):
                            state.call_id = tool_id_value
                        function_payload = tool.get("function", {})
                        arguments_payload = function_payload.get("arguments")
                        if isinstance(arguments_payload, str) and arguments_payload:
                            state.final_arguments = arguments_payload
                            state.arguments = arguments_payload
                        if state.item_id is None:
                            state.item_id = call_identifier

                    for state in tool_tracker.values():
                        if state.item_id is None:
                            state.item_id = (
                                state.call_id or state.id or f"call_{state.index}"
                            )
                        for event in emit_anthropic_tool_use_events(
                            current_index,
                            state,
                            parser=_parse_tool_input,
                        ):
                            yield event
                        current_index += 1

                    usage_payload = chunk_payload.get("usage")
                    if usage_payload is None and not isinstance(chunk, dict):
                        usage_obj = getattr(chunk, "usage", None)
                        if usage_obj is not None:
                            if hasattr(usage_obj, "model_dump"):
                                usage_payload = usage_obj.model_dump()
                            else:
                                usage_payload = {
                                    "prompt_tokens": getattr(
                                        usage_obj, "prompt_tokens", 0
                                    ),
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


async def convert__openai_responses_to_anthropic_messages__stream(
    stream: AsyncIterator[Any],
) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
    """Translate OpenAI Responses streaming events into Anthropic message events."""

    adapter = OpenAIResponsesToAnthropicStreamAdapter()
    async for event in adapter.run(stream):
        yield event


def convert__openai_chat_to_anthropic_messages__stream(
    stream: AsyncIterator[Any],
) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
    """Translate OpenAI ChatCompletion streams into Anthropic message events."""

    adapter = OpenAIChatToAnthropicStreamAdapter()
    return adapter.run(stream)


__all__ = [
    "OpenAIChatToAnthropicStreamAdapter",
    "OpenAIResponsesToAnthropicStreamAdapter",
    "convert__openai_chat_to_anthropic_messages__stream",
    "convert__openai_responses_to_anthropic_messages__stream",
]
