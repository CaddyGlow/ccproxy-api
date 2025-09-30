"""Streaming conversion entry points for OpenAI↔OpenAI adapters."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal

from pydantic import ValidationError

import ccproxy.core.logging
from ccproxy.llms.formatters.common import (
    THINKING_CLOSE_PATTERN,
    THINKING_OPEN_PATTERN,
    IndexedToolCallTracker,
    ObfuscationTokenFactory,
    ReasoningBuffer,
    ThinkingSegment,
    ToolCallState,
    ToolCallTracker,
    ensure_identifier,
)
from ccproxy.llms.formatters.context import (
    get_last_instructions,
    get_last_request,
    get_last_request_tools,
    register_request,
    register_request_tools,
)
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import OpenAIAccumulator

from ._helpers import (
    _convert_tools_chat_to_responses,
    _get_attr,
)
from .requests import _build_responses_payload_from_chat_request
from .responses import (
    _collect_reasoning_segments,
    _wrap_thinking,
    convert__openai_completion_usage_to_openai_responses__usage,
    convert__openai_responses_usage_to_openai_completion__usage,
)


logger = ccproxy.core.logging.get_logger(__name__)


class OpenAIResponsesToChatStreamAdapter:
    """Stateful adapter for Responses -> Chat streaming conversions."""

    def run(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        """Convert Response API stream events to ChatCompletionChunk events."""

        async def generator() -> AsyncGenerator[
            openai_models.ChatCompletionChunk, None
        ]:
            model_id = ""
            role_sent = False

            # Track tool call state keyed by response item id
            tool_tracker = ToolCallTracker()
            tool_delta_emitted = False
            saw_tool_event = False
            tool_candidates: list[tuple[str | None, set[str]]] = []
            reasoning_buffer = ReasoningBuffer()

            def _extract_tool_signature(tool_entry: Any) -> tuple[str | None, set[str]]:
                name: str | None = None
                param_keys: set[str] = set()

                if hasattr(tool_entry, "function"):
                    fn = getattr(tool_entry, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", None)
                        parameters = getattr(fn, "parameters", None)
                        if isinstance(parameters, dict):
                            props = parameters.get("properties")
                            if isinstance(props, dict):
                                param_keys = {str(key) for key in props}
                if name is None and isinstance(tool_entry, dict):
                    fn_dict = tool_entry.get("function")
                    if isinstance(fn_dict, dict):
                        name = fn_dict.get("name", name)
                        parameters = fn_dict.get("parameters")
                        if isinstance(parameters, dict):
                            props = parameters.get("properties")
                            if isinstance(props, dict):
                                param_keys = {str(key) for key in props}
                    if name is None:
                        name = tool_entry.get("name")

                return name, param_keys

            def _guess_tool_name(arguments: str | None) -> str | None:
                if not arguments:
                    return None
                try:
                    parsed = json.loads(arguments)
                except Exception:
                    return None
                if not isinstance(parsed, dict):
                    return None
                keys = {str(k) for k in parsed}
                if not keys:
                    return None

                candidates = [
                    tool_name
                    for tool_name, param_keys in tool_candidates
                    if tool_name
                    and ((param_keys and keys.issubset(param_keys)) or not param_keys)
                ]

                if len(candidates) == 1:
                    return candidates[0]

                exact = [
                    tool_name
                    for tool_name, param_keys in tool_candidates
                    if tool_name and param_keys == keys
                ]
                if len(exact) == 1:
                    return exact[0]

                return None

            def _ensure_tool_state(item_id: str) -> ToolCallState:
                return tool_tracker.ensure(item_id)

            item_id = "msg_stream"
            output_index = 0
            content_index = 0
            sequence_counter = 0
            first_logged = False

            inline_reasoning_id = "__inline_reasoning__"
            inline_summary_index = "__inline__"

            async for event_wrapper in stream:
                evt = getattr(event_wrapper, "root", event_wrapper)
                if not hasattr(evt, "type"):
                    continue

                logger.debug("stream_event", event_type=getattr(evt, "type", None))
                evt_type = getattr(evt, "type", "")

                if evt_type == "response.reasoning_summary_part.added":
                    item_id = _get_attr(evt, "item_id")
                    part = _get_attr(evt, "part")
                    if isinstance(item_id, str) and item_id and part is not None:
                        summary_index = _get_attr(evt, "summary_index")
                        part_signature = _get_attr(part, "signature")
                        if isinstance(part_signature, str) and part_signature:
                            reasoning_buffer.set_signature(
                                item_id, summary_index, part_signature
                            )
                        else:
                            part_type = _get_attr(part, "type")
                            part_text = _get_attr(part, "text")
                            if (
                                part_type == "signature"
                                and isinstance(part_text, str)
                                and part_text
                            ):
                                reasoning_buffer.set_signature(
                                    item_id, summary_index, part_text
                                )
                        reasoning_buffer.reset_buffer(item_id, summary_index)
                    continue

                if evt_type in {
                    "response.reasoning_summary_text.delta",
                    "response.reasoning_text.delta",
                }:
                    item_id = _get_attr(evt, "item_id")
                    delta_text = _get_attr(evt, "delta")
                    if isinstance(item_id, str):
                        summary_index = _get_attr(evt, "summary_index")
                        reasoning_buffer.append_text(item_id, summary_index, delta_text)
                    continue

                if evt_type in {
                    "response.reasoning_summary_text.done",
                    "response.reasoning_text.done",
                }:
                    item_id = _get_attr(evt, "item_id")
                    text_value = _get_attr(evt, "text")
                    if isinstance(item_id, str):
                        summary_index = _get_attr(evt, "summary_index")
                        for chunk_text in reasoning_buffer.emit(
                            item_id, summary_index, text_value
                        ):
                            sequence_counter += 1
                            yield openai_models.ChatCompletionChunk(
                                id="chatcmpl-stream",
                                created=0,
                                model=model_id,
                                choices=[
                                    openai_models.StreamingChoice(
                                        index=0,
                                        delta=openai_models.DeltaMessage(
                                            role="assistant" if not role_sent else None,
                                            content=chunk_text,
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            role_sent = True
                    continue

                if evt_type == "response.created":
                    response_obj = getattr(evt, "response", None)
                    model_id = getattr(response_obj, "model", model_id) or model_id
                    tools_metadata = getattr(response_obj, "tools", None)
                    if not tools_metadata:
                        tools_metadata = get_last_request_tools() or []
                    if tools_metadata:
                        tool_candidates = [
                            _extract_tool_signature(entry) for entry in tools_metadata
                        ]
                    continue

                if evt_type == "response.output_text.delta":
                    delta_text = getattr(evt, "delta", None) or ""
                    if not delta_text:
                        continue

                    remaining = delta_text

                    # Directly create chunks and yield them instead of using a nested function
                    # which has closure binding issues
                    chunks_to_yield: list[openai_models.ChatCompletionChunk] = []

                    def create_text_chunk(
                        current_model_id: str, text_segment: str, is_role_sent: bool
                    ) -> tuple[openai_models.ChatCompletionChunk | None, bool]:
                        if not text_segment:
                            return None, is_role_sent
                        delta_msg = openai_models.DeltaMessage(
                            role="assistant" if not is_role_sent else None,
                            content=text_segment,
                        )
                        new_role_sent = True
                        chunk = openai_models.ChatCompletionChunk(
                            id="chatcmpl-stream",
                            created=0,
                            model=current_model_id,
                            choices=[
                                openai_models.StreamingChoice(
                                    index=0,
                                    delta=delta_msg,
                                    finish_reason=None,
                                )
                            ],
                        )
                        return chunk, new_role_sent

                    while remaining:
                        if reasoning_buffer.is_open(
                            inline_reasoning_id, inline_summary_index
                        ):
                            close_match = THINKING_CLOSE_PATTERN.search(remaining)
                            if close_match:
                                inside_text = remaining[: close_match.start()]
                                if inside_text:
                                    reasoning_buffer.append_text(
                                        inline_reasoning_id,
                                        inline_summary_index,
                                        inside_text,
                                    )
                                for chunk_text in reasoning_buffer.emit(
                                    inline_reasoning_id, inline_summary_index
                                ):
                                    chunk, role_sent = create_text_chunk(
                                        model_id, chunk_text, role_sent
                                    )
                                    if chunk:
                                        sequence_counter += 1
                                        chunks_to_yield.append(chunk)
                                reasoning_buffer.close_part(
                                    inline_reasoning_id, inline_summary_index
                                )
                                remaining = remaining[close_match.end() :]
                                continue
                            reasoning_buffer.append_text(
                                inline_reasoning_id,
                                inline_summary_index,
                                remaining,
                            )
                            remaining = ""
                            break

                        open_match = THINKING_OPEN_PATTERN.search(remaining)
                        if open_match:
                            prefix_text = remaining[: open_match.start()]
                            if prefix_text:
                                chunk, role_sent = create_text_chunk(
                                    model_id, prefix_text, role_sent
                                )
                                if chunk:
                                    sequence_counter += 1
                                    chunks_to_yield.append(chunk)

                            signature = open_match.group(1) or None
                            part_state = reasoning_buffer.ensure_part(
                                inline_reasoning_id, inline_summary_index
                            )
                            if signature:
                                part_state.signature = signature
                            remaining = remaining[open_match.end() :]

                            if part_state.open:
                                # Already inside a reasoning block; ignore duplicate tag
                                continue

                            reasoning_buffer.open_part(
                                inline_reasoning_id, inline_summary_index
                            )
                            continue

                        # No reasoning markers in the rest of the chunk
                        if reasoning_buffer.is_open(
                            inline_reasoning_id, inline_summary_index
                        ):
                            reasoning_buffer.append_text(
                                inline_reasoning_id, inline_summary_index, remaining
                            )
                        else:
                            chunk, role_sent = create_text_chunk(
                                model_id, remaining, role_sent
                            )
                            if chunk:
                                sequence_counter += 1
                                chunks_to_yield.append(chunk)
                        remaining = ""

                    for chunk in chunks_to_yield:
                        yield chunk
                    continue

                if evt_type == "response.output_item.added":
                    item = getattr(evt, "item", None)
                    if not item:
                        continue

                    item_type = getattr(item, "type", None)
                    if item_type != "function_call":
                        continue

                    saw_tool_event = True

                    item_id_value = getattr(item, "id", None) or getattr(
                        item, "call_id", None
                    )
                    if not item_id_value:
                        item_id_value = f"call_{uuid.uuid4().hex}"
                    item_id = item_id_value

                    state = _ensure_tool_state(item_id)
                    state.id = getattr(item, "id", state.id) or state.id
                    state.call_id = getattr(item, "call_id", None) or state.call_id

                    if not state.name and state.index < len(tool_candidates):
                        candidate_name = tool_candidates[state.index][0]
                        if candidate_name:
                            state.name = candidate_name

                    name = getattr(item, "name", None)
                    if name:
                        state.name = name

                    arguments = getattr(item, "arguments", None)
                    if isinstance(arguments, str) and arguments:
                        state.arguments += arguments
                        if not state.name:
                            guessed = _guess_tool_name(state.arguments)
                            if guessed:
                                state.name = guessed

                    # Emit initial tool call chunk to surface id/name information
                    if not state.initial_emitted:
                        tool_call = openai_models.ToolCall(
                            id=state.id,
                            type="function",
                            function=openai_models.FunctionCall(
                                name=state.name or "",
                                arguments=arguments or "",
                            ),
                        )
                        state.emitted = True
                        state.initial_emitted = True
                        if state.name:
                            state.name_emitted = True
                        if arguments:
                            state.arguments_emitted = True

                        tool_delta_emitted = True

                        yield openai_models.ChatCompletionChunk(
                            id="chatcmpl-stream",
                            created=0,
                            model=model_id,
                            choices=[
                                openai_models.StreamingChoice(
                                    index=0,
                                    delta=openai_models.DeltaMessage(
                                        role="assistant" if not role_sent else None,
                                        tool_calls=[tool_call],
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        role_sent = True
                    continue

                if evt_type == "response.function_call_arguments.delta":
                    saw_tool_event = True
                    item_id_val = getattr(evt, "item_id", None)
                    if not isinstance(item_id_val, str):
                        continue
                    item_id = item_id_val
                    delta_segment = getattr(evt, "delta", None)
                    if not isinstance(delta_segment, str):
                        continue

                    state = _ensure_tool_state(item_id)
                    state.arguments += delta_segment
                    if not state.name:
                        guessed = _guess_tool_name(state.arguments)
                        if guessed:
                            state.name = guessed

                    if state.initial_emitted:
                        tool_call = openai_models.ToolCall(
                            id=state.id,
                            type="function",
                            function=openai_models.FunctionCall(
                                name=state.name or "",
                                arguments=delta_segment,
                            ),
                        )

                        state.emitted = True
                        if delta_segment:
                            state.arguments_emitted = True

                        tool_delta_emitted = True

                        yield openai_models.ChatCompletionChunk(
                            id="chatcmpl-stream",
                            created=0,
                            model=model_id,
                            choices=[
                                openai_models.StreamingChoice(
                                    index=0,
                                    delta=openai_models.DeltaMessage(
                                        role="assistant" if not role_sent else None,
                                        tool_calls=[tool_call],
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        role_sent = True
                    continue

                if evt_type == "response.function_call_arguments.done":
                    saw_tool_event = True
                    item_id_val = getattr(evt, "item_id", None)
                    if not isinstance(item_id_val, str):
                        continue
                    item_id = item_id_val
                    arguments = getattr(evt, "arguments", None)
                    if not isinstance(arguments, str) or not arguments:
                        continue

                    state = _ensure_tool_state(item_id)
                    # Only emit a chunk if we never emitted arguments earlier
                    if not state.arguments_emitted:
                        state.arguments = arguments
                        if not state.name:
                            guessed = _guess_tool_name(arguments)
                            if guessed:
                                state.name = guessed

                        tool_call = openai_models.ToolCall(
                            id=state.id,
                            type="function",
                            function=openai_models.FunctionCall(
                                name=state.name or "",
                                arguments=arguments,
                            ),
                        )

                        state.emitted = True
                        state.arguments_emitted = True

                        tool_delta_emitted = True

                        yield openai_models.ChatCompletionChunk(
                            id="chatcmpl-stream",
                            created=0,
                            model=model_id,
                            choices=[
                                openai_models.StreamingChoice(
                                    index=0,
                                    delta=openai_models.DeltaMessage(
                                        role="assistant" if not role_sent else None,
                                        tool_calls=[tool_call],
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        role_sent = True
                    continue

                if evt_type == "response.output_item.done":
                    item = getattr(evt, "item", None)
                    if not item:
                        continue

                    item_type = getattr(item, "type", None)

                    if item_type == "reasoning":
                        summary_list = getattr(item, "summary", None)
                        if isinstance(summary_list, list):
                            for entry in summary_list:
                                text = _get_attr(entry, "text")
                                signature = _get_attr(entry, "signature")
                                if isinstance(text, str) and text:
                                    chunk_text = _wrap_thinking(signature, text)
                                    sequence_counter += 1
                                    yield openai_models.ChatCompletionChunk(
                                        id="chatcmpl-stream",
                                        created=0,
                                        model=model_id,
                                        choices=[
                                            openai_models.StreamingChoice(
                                                index=0,
                                                delta=openai_models.DeltaMessage(
                                                    role="assistant"
                                                    if not role_sent
                                                    else None,
                                                    content=chunk_text,
                                                ),
                                                finish_reason=None,
                                            )
                                        ],
                                    )
                                    role_sent = True
                        continue

                    if item_type != "function_call":
                        continue

                    saw_tool_event = True

                    item_id_value = getattr(item, "id", None) or getattr(
                        item, "call_id", None
                    )
                    if not isinstance(item_id_value, str) or not item_id_value:
                        continue
                    item_id = item_id_value

                    state = _ensure_tool_state(item_id)
                    name = getattr(item, "name", None)
                    if name:
                        state.name = name
                    arguments = getattr(item, "arguments", None)
                    if isinstance(arguments, str) and arguments:
                        state.arguments = arguments
                        if not state.name:
                            guessed = _guess_tool_name(arguments)
                            if guessed:
                                state.name = guessed
                        if not state.arguments_emitted:
                            tool_call = openai_models.ToolCall(
                                id=state.id,
                                type="function",
                                function=openai_models.FunctionCall(
                                    name=state.name or "",
                                    arguments=arguments,
                                ),
                            )
                            state.emitted = True
                            state.arguments_emitted = True

                            yield openai_models.ChatCompletionChunk(
                                id="chatcmpl-stream",
                                created=0,
                                model=model_id,
                                choices=[
                                    openai_models.StreamingChoice(
                                        index=0,
                                        delta=openai_models.DeltaMessage(
                                            role="assistant" if not role_sent else None,
                                            tool_calls=[tool_call],
                                        ),
                                        finish_reason=None,
                                    )
                                ],
                            )
                            role_sent = True

                    # Emit a patch chunk if the name was never surfaced earlier
                    if state.name and not state.name_emitted:
                        tool_call = openai_models.ToolCall(
                            id=state.id,
                            type="function",
                            function=openai_models.FunctionCall(
                                name=state.name or "",
                                arguments="",
                            ),
                        )
                        state.name_emitted = True

                        tool_delta_emitted = True

                        yield openai_models.ChatCompletionChunk(
                            id="chatcmpl-stream",
                            created=0,
                            model=model_id,
                            choices=[
                                openai_models.StreamingChoice(
                                    index=0,
                                    delta=openai_models.DeltaMessage(
                                        role="assistant" if not role_sent else None,
                                        tool_calls=[tool_call],
                                    ),
                                    finish_reason=None,
                                )
                            ],
                        )
                        role_sent = True

                    state.completed = True
                    continue

                if evt_type in {
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                }:
                    usage = None
                    response_obj = getattr(evt, "response", None)
                    if response_obj and getattr(response_obj, "usage", None):
                        usage = (
                            convert__openai_responses_usage_to_openai_completion__usage(
                                response_obj.usage
                            )
                        )

                    finish_reason: Literal["stop", "length", "tool_calls"] = "stop"
                    if (
                        tool_delta_emitted
                        or saw_tool_event
                        or len(tool_tracker)
                        or tool_tracker.any_completed()
                    ):
                        finish_reason = "tool_calls"

                    yield openai_models.ChatCompletionChunk(
                        id="chatcmpl-stream",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(),
                                finish_reason=finish_reason,
                            )
                        ],
                        usage=usage,
                    )

                    # Cleanup request tool cache context when stream completes
                    register_request_tools(None)

        return generator()


def convert__openai_responses_to_openai_chat__stream(
    stream: AsyncIterator[openai_models.AnyStreamEvent],
) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
    """Convert Response API stream events to ChatCompletionChunk events."""
    adapter = OpenAIResponsesToChatStreamAdapter()
    return adapter.run(stream)


class OpenAIChatToResponsesStreamAdapter:
    """Stateful adapter for Chat -> Responses streaming conversions."""

    def run(
        self,
        stream: AsyncIterator[openai_models.ChatCompletionChunk | dict[str, Any]],
    ) -> AsyncGenerator[openai_models.StreamEventType, None]:
        """Convert OpenAI ChatCompletionChunk stream to Responses API events.

        Replays chat deltas as Responses events, including function-call output items
        and argument deltas so partial tool calls stream correctly.
        """

        async def generator() -> AsyncGenerator[openai_models.StreamEventType, None]:
            log = logger.bind(
                category="formatter", converter="chat_to_responses_stream"
            )

            created_sent = False
            response_id = ""
            id_suffix: str | None = None
            last_model = ""
            sequence_counter = -1
            first_logged = False

            openai_accumulator = OpenAIAccumulator()
            latest_usage_model: openai_models.ResponseUsage | None = None
            convert_usage = convert__openai_completion_usage_to_openai_responses__usage
            delta_event_cls = openai_models.ResponseFunctionCallArgumentsDeltaEvent

            instructions_text = get_last_instructions()
            if not instructions_text:
                try:
                    from ccproxy.core.request_context import RequestContext

                    ctx = RequestContext.get_current()
                    if ctx is not None:
                        raw_instr = ctx.metadata.get("instructions")
                        if isinstance(raw_instr, str) and raw_instr.strip():
                            instructions_text = raw_instr.strip()
                except Exception:
                    pass
            instructions_value = instructions_text or None

            envelope_base_kwargs: dict[str, Any] = {
                "id": response_id,
                "object": "response",
                "created_at": 0,
                "instructions": instructions_value,
            }
            reasoning_summary_payload: list[dict[str, Any]] | None = None

            last_request = get_last_request()
            chat_request: openai_models.ChatCompletionRequest | None = None
            if isinstance(last_request, openai_models.ChatCompletionRequest):
                chat_request = last_request
            elif isinstance(last_request, dict):
                try:
                    chat_request = openai_models.ChatCompletionRequest.model_validate(
                        last_request
                    )
                except ValidationError:
                    chat_request = None

            base_parallel_tool_calls = True
            text_payload: dict[str, Any] | None = None

            if chat_request is not None:
                request_payload, _ = _build_responses_payload_from_chat_request(
                    chat_request
                )
                base_parallel_tool_calls = bool(
                    request_payload.get("parallel_tool_calls", True)
                )
                background_value = request_payload.get("background", None)
                envelope_base_kwargs["background"] = (
                    bool(background_value) if background_value is not None else None
                )
                for key in (
                    "max_output_tokens",
                    "tool_choice",
                    "tools",
                    "store",
                    "service_tier",
                    "temperature",
                    "prompt_cache_key",
                    "top_p",
                    "top_logprobs",
                    "truncation",
                    "metadata",
                    "user",
                ):
                    if key in request_payload:
                        envelope_base_kwargs[key] = request_payload[key]
                text_payload = request_payload.get("text")
                reasoning_source = request_payload.get("reasoning")
                reasoning_effort = None
                if isinstance(reasoning_source, dict):
                    reasoning_effort = reasoning_source.get("effort")
                if reasoning_effort is None:
                    reasoning_effort = getattr(chat_request, "reasoning_effort", None)
                envelope_base_kwargs["reasoning"] = openai_models.Reasoning(
                    effort=reasoning_effort,
                    summary=None,
                )
                if envelope_base_kwargs.get("tool_choice") is None:
                    envelope_base_kwargs["tool_choice"] = (
                        chat_request.tool_choice or "auto"
                    )
                if envelope_base_kwargs.get("tools") is None and chat_request.tools:
                    envelope_base_kwargs["tools"] = _convert_tools_chat_to_responses(
                        chat_request.tools
                    )
                if envelope_base_kwargs.get("store") is None:
                    store_value = getattr(chat_request, "store", None)
                    if store_value is not None:
                        envelope_base_kwargs["store"] = store_value
                if envelope_base_kwargs.get("temperature") is None:
                    temperature_value = getattr(chat_request, "temperature", None)
                    if temperature_value is not None:
                        envelope_base_kwargs["temperature"] = temperature_value
                if envelope_base_kwargs.get("service_tier") is None:
                    service_tier_value = getattr(chat_request, "service_tier", None)
                    envelope_base_kwargs["service_tier"] = service_tier_value or "auto"
                if "metadata" not in envelope_base_kwargs:
                    envelope_base_kwargs["metadata"] = {}
                register_request_tools(chat_request.tools)
            else:
                envelope_base_kwargs["background"] = envelope_base_kwargs.get(
                    "background"
                )
                envelope_base_kwargs["reasoning"] = openai_models.Reasoning(
                    effort=None, summary=None
                )
                envelope_base_kwargs.setdefault("metadata", {})

            if text_payload is None:
                text_payload = {"format": {"type": "text"}}
            else:
                text_payload = dict(text_payload)

            verbosity_value = None
            if chat_request is not None:
                verbosity_value = getattr(chat_request, "verbosity", None)
            if verbosity_value is not None:
                text_payload["verbosity"] = verbosity_value
            else:
                text_payload.setdefault("verbosity", "low")
            envelope_base_kwargs["text"] = text_payload

            if "store" not in envelope_base_kwargs:
                envelope_base_kwargs["store"] = True
            if "temperature" not in envelope_base_kwargs:
                envelope_base_kwargs["temperature"] = 1.0
            if "service_tier" not in envelope_base_kwargs:
                envelope_base_kwargs["service_tier"] = "auto"
            if "tool_choice" not in envelope_base_kwargs:
                envelope_base_kwargs["tool_choice"] = "auto"
            if "prompt_cache_key" not in envelope_base_kwargs:
                envelope_base_kwargs["prompt_cache_key"] = None
            if "top_p" not in envelope_base_kwargs:
                envelope_base_kwargs["top_p"] = 1.0
            if "top_logprobs" not in envelope_base_kwargs:
                envelope_base_kwargs["top_logprobs"] = None
            if "truncation" not in envelope_base_kwargs:
                envelope_base_kwargs["truncation"] = None
            if "user" not in envelope_base_kwargs:
                envelope_base_kwargs["user"] = None

            parallel_setting_initial = bool(base_parallel_tool_calls)
            envelope_base_kwargs["parallel_tool_calls"] = parallel_setting_initial

            message_item_id = ""
            message_output_index: int | None = None
            content_index = 0
            message_item_added = False
            message_content_part_added = False
            message_text_buffer: list[str] = []
            message_last_logprobs: Any | None = None
            message_text_done_emitted = False
            message_part_done_emitted = False
            message_item_done_emitted = False
            message_completed_entry: tuple[int, openai_models.MessageOutput] | None = (
                None
            )

            reasoning_item_id = ""
            reasoning_output_index: int | None = None
            reasoning_item_added = False
            reasoning_output_done = False
            reasoning_summary_indices: dict[str, int] = {}
            reasoning_summary_added: set[int] = set()
            reasoning_summary_text_fragments: dict[int, list[str]] = {}
            reasoning_summary_text_done: set[int] = set()
            reasoning_summary_part_done: set[int] = set()
            reasoning_completed_entry: (
                tuple[int, openai_models.ReasoningOutput] | None
            ) = None
            next_summary_index = 0
            reasoning_summary_signatures: dict[int, str | None] = {}

            created_at_value: int | None = None

            next_output_index = 0
            tool_call_states = IndexedToolCallTracker()

            obfuscation_factory = ObfuscationTokenFactory(
                lambda: id_suffix or response_id or "stream"
            )

            def ensure_message_output_item() -> (
                openai_models.ResponseOutputItemAddedEvent | None
            ):
                nonlocal message_item_added, message_output_index, next_output_index
                nonlocal sequence_counter
                if message_output_index is None:
                    message_output_index = next_output_index
                    next_output_index += 1
                if not message_item_added:
                    message_item_added = True
                    sequence_counter += 1
                    return openai_models.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=sequence_counter,
                        output_index=message_output_index,
                        item=openai_models.OutputItem(
                            id=message_item_id,
                            type="message",
                            role="assistant",
                            status="in_progress",
                            content=[],
                        ),
                    )
                return None

            def ensure_message_content_part() -> (
                openai_models.ResponseContentPartAddedEvent | None
            ):
                nonlocal message_content_part_added, sequence_counter
                if message_output_index is None:
                    return None
                if not message_content_part_added:
                    message_content_part_added = True
                    sequence_counter += 1
                    return openai_models.ResponseContentPartAddedEvent(
                        type="response.content_part.added",
                        sequence_number=sequence_counter,
                        item_id=message_item_id,
                        output_index=message_output_index,
                        content_index=content_index,
                        part=openai_models.ContentPart(
                            type="output_text",
                            text="",
                            annotations=[],
                        ),
                    )
                return None

            def emit_message_text_delta(
                delta_text: str,
                *,
                logprobs: Any | None = None,
                obfuscation: str | None = None,
            ) -> list[openai_models.StreamEventType]:
                if not isinstance(delta_text, str) or not delta_text:
                    return []

                nonlocal \
                    message_last_logprobs, \
                    sequence_counter, \
                    message_item_done_emitted
                if message_item_done_emitted:
                    return []

                events: list[openai_models.StreamEventType] = []

                message_event = ensure_message_output_item()
                if message_event is not None:
                    events.append(message_event)

                content_event = ensure_message_content_part()
                if content_event is not None:
                    events.append(content_event)

                sequence_counter += 1
                event_sequence = sequence_counter
                logprobs_value: Any
                if logprobs is None:
                    logprobs_value = []
                else:
                    logprobs_value = logprobs
                obfuscation_value = obfuscation or obfuscation_factory.make(
                    "message.delta",
                    sequence=event_sequence,
                    item_id=message_item_id,
                    payload=delta_text,
                )
                events.append(
                    openai_models.ResponseOutputTextDeltaEvent(
                        type="response.output_text.delta",
                        sequence_number=event_sequence,
                        item_id=message_item_id,
                        output_index=message_output_index or 0,
                        content_index=content_index,
                        delta=delta_text,
                        logprobs=logprobs_value,
                    )
                )
                message_text_buffer.append(delta_text)
                message_last_logprobs = logprobs_value
                return events

            def _reasoning_key(signature: str | None) -> str:
                if isinstance(signature, str) and signature.strip():
                    return signature.strip()
                return "__default__"

            def get_summary_index(signature: str | None) -> int:
                nonlocal next_summary_index
                key = _reasoning_key(signature)
                maybe_index = reasoning_summary_indices.get(key)
                if maybe_index is not None:
                    return maybe_index
                reasoning_summary_indices[key] = next_summary_index
                next_summary_index += 1
                return reasoning_summary_indices[key]

            def ensure_reasoning_output_item() -> (
                openai_models.ResponseOutputItemAddedEvent | None
            ):
                nonlocal reasoning_item_added, reasoning_output_index
                nonlocal next_output_index, sequence_counter
                if reasoning_output_index is None:
                    reasoning_output_index = next_output_index
                    next_output_index += 1
                if not reasoning_item_added:
                    reasoning_item_added = True
                    sequence_counter += 1
                    return openai_models.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=sequence_counter,
                        output_index=reasoning_output_index,
                        item=openai_models.OutputItem(
                            id=reasoning_item_id,
                            type="reasoning",
                            status="in_progress",
                            summary=[],
                        ),
                    )
                return None

            def ensure_reasoning_summary_part(
                summary_index: int,
            ) -> openai_models.ReasoningSummaryPartAddedEvent | None:
                nonlocal sequence_counter
                if reasoning_output_index is None:
                    return None
                if summary_index in reasoning_summary_added:
                    return None
                reasoning_summary_added.add(summary_index)
                sequence_counter += 1
                return openai_models.ReasoningSummaryPartAddedEvent(
                    type="response.reasoning_summary_part.added",
                    sequence_number=sequence_counter,
                    item_id=reasoning_item_id,
                    output_index=reasoning_output_index,
                    summary_index=summary_index,
                    part=openai_models.ReasoningSummaryPart(
                        type="summary_text",
                        text="",
                    ),
                )

            def emit_reasoning_segments(
                segments: list[ThinkingSegment],
            ) -> list[openai_models.StreamEventType]:
                events: list[openai_models.StreamEventType] = []
                if not segments:
                    return events

                output_event = ensure_reasoning_output_item()
                if output_event is not None:
                    events.append(output_event)

                nonlocal sequence_counter
                for segment in segments:
                    text_value = getattr(segment, "thinking", "")
                    if not isinstance(text_value, str) or not text_value:
                        continue
                    summary_index = get_summary_index(
                        getattr(segment, "signature", None)
                    )
                    signature_value = getattr(segment, "signature", None)
                    if summary_index not in reasoning_summary_signatures:
                        reasoning_summary_signatures[summary_index] = signature_value
                    part_event = ensure_reasoning_summary_part(summary_index)
                    if part_event is not None:
                        events.append(part_event)
                    fragments = reasoning_summary_text_fragments.setdefault(
                        summary_index, []
                    )
                    fragments.append(text_value)
                    sequence_counter += 1
                    event_sequence = sequence_counter
                    events.append(
                        openai_models.ReasoningSummaryTextDeltaEvent(
                            type="response.reasoning_summary_text.delta",
                            sequence_number=event_sequence,
                            item_id=reasoning_item_id,
                            output_index=reasoning_output_index or 0,
                            summary_index=summary_index,
                            delta=text_value,
                        )
                    )
                return events

            def finalize_reasoning() -> list[openai_models.StreamEventType]:
                nonlocal reasoning_output_done, reasoning_completed_entry
                nonlocal reasoning_summary_payload, sequence_counter
                if not reasoning_item_added or reasoning_output_index is None:
                    return []

                events: list[openai_models.StreamEventType] = []
                summary_entries: list[dict[str, Any]] = []

                for summary_index in sorted(reasoning_summary_text_fragments):
                    text_value = "".join(
                        reasoning_summary_text_fragments.get(summary_index, [])
                    )
                    if summary_index not in reasoning_summary_text_done:
                        sequence_counter += 1
                        events.append(
                            openai_models.ReasoningSummaryTextDoneEvent(
                                type="response.reasoning_summary_text.done",
                                sequence_number=sequence_counter,
                                item_id=reasoning_item_id,
                                output_index=reasoning_output_index,
                                summary_index=summary_index,
                                text=text_value,
                            )
                        )
                        reasoning_summary_text_done.add(summary_index)
                    if summary_index not in reasoning_summary_part_done:
                        sequence_counter += 1
                        events.append(
                            openai_models.ReasoningSummaryPartDoneEvent(
                                type="response.reasoning_summary_part.done",
                                sequence_number=sequence_counter,
                                item_id=reasoning_item_id,
                                output_index=reasoning_output_index,
                                summary_index=summary_index,
                                part=openai_models.ReasoningSummaryPart(
                                    type="summary_text",
                                    text=text_value,
                                ),
                            )
                        )
                        reasoning_summary_part_done.add(summary_index)
                    summary_entry: dict[str, Any] = {
                        "type": "summary_text",
                        "text": text_value,
                    }
                    signature_value = reasoning_summary_signatures.get(summary_index)
                    if signature_value:
                        summary_entry["signature"] = signature_value
                    summary_entries.append(summary_entry)

                reasoning_summary_payload = summary_entries

                if not reasoning_output_done:
                    sequence_counter += 1
                    events.append(
                        openai_models.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=sequence_counter,
                            output_index=reasoning_output_index,
                            item=openai_models.OutputItem(
                                id=reasoning_item_id,
                                type="reasoning",
                                status="completed",
                                summary=summary_entries,
                            ),
                        )
                    )
                    reasoning_output_done = True
                    reasoning_completed_entry = (
                        reasoning_output_index,
                        openai_models.ReasoningOutput(
                            type="reasoning",
                            id=reasoning_item_id,
                            status="completed",
                            summary=summary_entries,
                        ),
                    )

                return events

            def finalize_message() -> list[openai_models.StreamEventType]:
                nonlocal sequence_counter
                nonlocal message_text_done_emitted, message_part_done_emitted
                nonlocal message_item_done_emitted, message_completed_entry
                nonlocal message_last_logprobs

                if not message_item_added:
                    return []

                events: list[openai_models.StreamEventType] = []
                final_text = "".join(message_text_buffer)
                logprobs_value: Any
                if message_last_logprobs is None:
                    logprobs_value = []
                else:
                    logprobs_value = message_last_logprobs

                if message_content_part_added and not message_text_done_emitted:
                    sequence_counter += 1
                    event_sequence = sequence_counter
                    events.append(
                        openai_models.ResponseOutputTextDoneEvent(
                            type="response.output_text.done",
                            sequence_number=event_sequence,
                            item_id=message_item_id,
                            output_index=message_output_index or 0,
                            content_index=content_index,
                            text=final_text,
                            logprobs=logprobs_value,
                        )
                    )
                    message_text_done_emitted = True

                if message_content_part_added and not message_part_done_emitted:
                    sequence_counter += 1
                    event_sequence = sequence_counter
                    events.append(
                        openai_models.ResponseContentPartDoneEvent(
                            type="response.content_part.done",
                            sequence_number=event_sequence,
                            item_id=message_item_id,
                            output_index=message_output_index or 0,
                            content_index=content_index,
                            part=openai_models.ContentPart(
                                type="output_text",
                                text=final_text,
                                annotations=[],
                            ),
                        )
                    )
                    message_part_done_emitted = True

                if not message_item_done_emitted:
                    sequence_counter += 1
                    event_sequence = sequence_counter
                    output_text_part = openai_models.OutputTextContent(
                        type="output_text",
                        text=final_text,
                        annotations=[],
                        logprobs=logprobs_value if logprobs_value != [] else [],
                    )
                    message_output = openai_models.MessageOutput(
                        type="message",
                        id=message_item_id,
                        status="completed",
                        role="assistant",
                        content=[output_text_part] if final_text else [],
                    )
                    message_completed_entry = (
                        message_output_index or 0,
                        message_output,
                    )
                    events.append(
                        openai_models.ResponseOutputItemDoneEvent(
                            type="response.output_item.done",
                            sequence_number=event_sequence,
                            output_index=message_output_index or 0,
                            item=openai_models.OutputItem(
                                id=message_item_id,
                                type="message",
                                role="assistant",
                                status="completed",
                                content=[output_text_part.model_dump()]
                                if final_text
                                else [],
                                text=final_text or None,
                            ),
                        )
                    )
                    message_item_done_emitted = True
                elif message_completed_entry is None:
                    output_text_part = openai_models.OutputTextContent(
                        type="output_text",
                        text=final_text,
                        annotations=[],
                        logprobs=logprobs_value if logprobs_value != [] else [],
                    )
                    message_completed_entry = (
                        message_output_index or 0,
                        openai_models.MessageOutput(
                            type="message",
                            id=message_item_id,
                            status="completed",
                            role="assistant",
                            content=[output_text_part] if final_text else [],
                        ),
                    )

                return events

            def get_tool_state(index: int) -> ToolCallState:
                nonlocal next_output_index
                state = tool_call_states.ensure(index)
                if state.output_index < 0:
                    state.output_index = next_output_index
                    next_output_index += 1
                return state

            def get_accumulator_entry(idx: int) -> dict[str, Any] | None:
                for entry in openai_accumulator.tools.values():
                    if entry.get("index") == idx:
                        return entry
                return None

            def emit_tool_item_added(
                state: ToolCallState,
            ) -> list[openai_models.StreamEventType]:
                nonlocal sequence_counter
                if state.added_emitted:
                    return []
                if state.name is None:
                    return []
                if not state.item_id:
                    item_identifier = state.call_id
                    if not item_identifier:
                        item_identifier = f"call_{state.index}"
                    state.item_id = item_identifier
                sequence_counter += 1
                state.added_emitted = True
                return [
                    openai_models.ResponseOutputItemAddedEvent(
                        type="response.output_item.added",
                        sequence_number=sequence_counter,
                        output_index=state.output_index,
                        item=openai_models.OutputItem(
                            id=state.item_id,
                            type="function_call",
                            status="in_progress",
                            name=state.name,
                            arguments="",
                            call_id=state.call_id,
                        ),
                    )
                ]

            def finalize_tool_calls() -> list[openai_models.StreamEventType]:
                nonlocal sequence_counter
                events: list[openai_models.StreamEventType] = []
                for idx, state in tool_call_states.items():
                    accumulator_entry = get_accumulator_entry(idx)
                    if state.name is None and accumulator_entry is not None:
                        fn_name = accumulator_entry.get("function", {}).get("name")
                        if isinstance(fn_name, str) and fn_name:
                            state.name = fn_name
                    if state.call_id is None and accumulator_entry is not None:
                        call_identifier = accumulator_entry.get("id")
                        if isinstance(call_identifier, str) and call_identifier:
                            state.call_id = call_identifier
                    if not state.item_id:
                        candidate_id = None
                        if accumulator_entry is not None:
                            candidate_id = accumulator_entry.get("id")
                        state.item_id = (
                            candidate_id or state.call_id or f"call_{state.index}"
                        )
                    if not state.added_emitted:
                        events.extend(emit_tool_item_added(state))
                    final_args = state.final_arguments
                    if final_args is None:
                        combined = "".join(state.arguments_parts or [])
                        if not combined and accumulator_entry is not None:
                            combined = (
                                accumulator_entry.get("function", {}).get("arguments")
                                or ""
                            )
                        final_args = combined or ""
                    state.final_arguments = final_args
                    if not state.arguments_done_emitted:
                        sequence_counter += 1
                        events.append(
                            openai_models.ResponseFunctionCallArgumentsDoneEvent(
                                type="response.function_call_arguments.done",
                                sequence_number=sequence_counter,
                                item_id=state.item_id,
                                output_index=state.output_index,
                                arguments=final_args,
                            )
                        )
                        state.arguments_done_emitted = True
                    if not state.item_done_emitted:
                        sequence_counter += 1
                        events.append(
                            openai_models.ResponseOutputItemDoneEvent(
                                type="response.output_item.done",
                                sequence_number=sequence_counter,
                                output_index=state.output_index,
                                item=openai_models.OutputItem(
                                    id=state.item_id,
                                    type="function_call",
                                    status="completed",
                                    name=state.name,
                                    arguments=final_args,
                                    call_id=state.call_id,
                                ),
                            )
                        )
                        state.item_done_emitted = True
                return events

            def make_response_object(
                *,
                status: str,
                model: str | None,
                usage: openai_models.ResponseUsage | None = None,
                output: list[Any] | None = None,
                parallel_override: bool | None = None,
                reasoning_summary: list[dict[str, Any]] | None = None,
                extra: dict[str, Any] | None = None,
            ) -> openai_models.ResponseObject:
                payload = dict(envelope_base_kwargs)
                payload["status"] = status
                payload["model"] = model or payload.get("model") or ""
                payload["output"] = output or []
                payload["usage"] = usage
                payload.setdefault("object", "response")
                payload.setdefault("created_at", int(time.time()))
                if parallel_override is not None:
                    payload["parallel_tool_calls"] = parallel_override
                if reasoning_summary is not None:
                    reasoning_entry = payload.get("reasoning")
                    if isinstance(reasoning_entry, openai_models.Reasoning):
                        payload["reasoning"] = reasoning_entry.model_copy(
                            update={"summary": reasoning_summary}
                        )
                    elif isinstance(reasoning_entry, dict):
                        payload["reasoning"] = openai_models.Reasoning(
                            effort=reasoning_entry.get("effort"),
                            summary=reasoning_summary,
                        )
                    else:
                        payload["reasoning"] = openai_models.Reasoning(
                            effort=None,
                            summary=reasoning_summary,
                        )
                if extra:
                    payload.update(extra)
                return openai_models.ResponseObject(**payload)

            try:
                async for chunk in stream:
                    if isinstance(chunk, dict):
                        chunk_payload = chunk
                    else:
                        chunk_payload = chunk.model_dump(exclude_none=True)

                    openai_accumulator.accumulate("", chunk_payload)

                    model = chunk_payload.get("model") or last_model
                    choices = chunk_payload.get("choices") or []
                    usage_obj = chunk_payload.get("usage")

                    finish_reasons: list[str | None] = []
                    deltas: list[dict[str, Any]] = []
                    for choice in choices:
                        if not isinstance(choice, dict):
                            continue
                        finish_reasons.append(choice.get("finish_reason"))
                        delta_obj = choice.get("delta") or {}
                        if isinstance(delta_obj, dict):
                            deltas.append(delta_obj)

                    last_model = model
                    if model:
                        envelope_base_kwargs["model"] = model

                    first_delta_text = deltas[0].get("content") if deltas else None

                    if not first_logged:
                        first_logged = True
                        with contextlib.suppress(Exception):
                            log.debug(
                                "chat_stream_first_chunk",
                                typed=isinstance(chunk, dict) is False,
                                keys=(
                                    list(chunk.keys())
                                    if isinstance(chunk, dict)
                                    else None
                                ),
                                has_delta=bool(first_delta_text),
                                model=model,
                            )
                            if len(choices) == 0 and not model:
                                log.debug("chat_stream_ignoring_first_chunk")
                                continue

                    if not created_sent:
                        created_sent = True
                        response_id, id_suffix = ensure_identifier(
                            "resp", chunk_payload.get("id")
                        )
                        envelope_base_kwargs["id"] = response_id
                        envelope_base_kwargs.setdefault("object", "response")
                        if not message_item_id:
                            message_item_id = f"msg_{id_suffix}"
                        if not reasoning_item_id:
                            reasoning_item_id = f"rs_{id_suffix}"

                        created_at_value = chunk_payload.get(
                            "created"
                        ) or chunk_payload.get("created_at")
                        if created_at_value is None:
                            created_at_value = int(time.time())
                        envelope_base_kwargs["created_at"] = int(created_at_value)

                        if model:
                            envelope_base_kwargs["model"] = model
                        elif last_model:
                            envelope_base_kwargs.setdefault("model", last_model)

                        sequence_counter += 1
                        response_created = make_response_object(
                            status="in_progress",
                            model=model or last_model,
                            usage=None,
                            output=[],
                            parallel_override=parallel_setting_initial,
                        )
                        yield openai_models.ResponseCreatedEvent(
                            type="response.created",
                            sequence_number=sequence_counter,
                            response=response_created,
                        )
                        sequence_counter += 1
                        yield openai_models.ResponseInProgressEvent(
                            type="response.in_progress",
                            sequence_number=sequence_counter,
                            response=make_response_object(
                                status="in_progress",
                                model=model or last_model,
                                usage=latest_usage_model,
                                output=[],
                                parallel_override=parallel_setting_initial,
                            ),
                        )

                    for delta in deltas:
                        reasoning_payload = delta.get("reasoning")
                        if reasoning_payload is not None:
                            segments = _collect_reasoning_segments(reasoning_payload)
                            for event in emit_reasoning_segments(segments):
                                yield event

                        content_value = delta.get("content")
                        if isinstance(content_value, str) and content_value:
                            for event in emit_message_text_delta(content_value):
                                yield event
                        elif isinstance(content_value, dict):
                            part_type = content_value.get("type")
                            if part_type in {"reasoning", "thinking"}:
                                segments = _collect_reasoning_segments(content_value)
                                for event in emit_reasoning_segments(segments):
                                    yield event
                            else:
                                text_value = content_value.get("text")
                                if not isinstance(text_value, str) or not text_value:
                                    delta_text = content_value.get("delta")
                                    if isinstance(delta_text, str) and delta_text:
                                        text_value = delta_text
                                if isinstance(text_value, str) and text_value:
                                    for event in emit_message_text_delta(
                                        text_value,
                                        logprobs=content_value.get("logprobs"),
                                        obfuscation=content_value.get("obfuscation")
                                        or content_value.get("obfuscated"),
                                    ):
                                        yield event
                        elif isinstance(content_value, list):
                            for part in content_value:
                                if not isinstance(part, dict):
                                    continue
                                part_type = part.get("type")
                                if part_type in {"reasoning", "thinking"}:
                                    segments = _collect_reasoning_segments(part)
                                    for event in emit_reasoning_segments(segments):
                                        yield event
                                    continue
                                text_value = part.get("text")
                                if not isinstance(text_value, str) or not text_value:
                                    delta_text = part.get("delta")
                                    if isinstance(delta_text, str) and delta_text:
                                        text_value = delta_text
                                if (
                                    part_type
                                    in {"text", "output_text", "output_text_delta"}
                                    and isinstance(text_value, str)
                                    and text_value
                                ):
                                    for event in emit_message_text_delta(
                                        text_value,
                                        logprobs=part.get("logprobs"),
                                        obfuscation=part.get("obfuscation")
                                        or part.get("obfuscated"),
                                    ):
                                        yield event

                        tool_calls = delta.get("tool_calls") or []
                        if isinstance(tool_calls, list):
                            if tool_calls:
                                for event in finalize_message():
                                    yield event
                            for tool_call in tool_calls:
                                if not isinstance(tool_call, dict):
                                    continue
                                index_value = int(tool_call.get("index", 0))
                                state = get_tool_state(index_value)
                                tool_id = tool_call.get("id")
                                if isinstance(tool_id, str) and tool_id:
                                    state.call_id = tool_id
                                    if not state.added_emitted or state.item_id is None:
                                        state.item_id = tool_id
                                function_obj = tool_call.get("function") or {}
                                if isinstance(function_obj, dict):
                                    name_value = function_obj.get("name")
                                    if isinstance(name_value, str) and name_value:
                                        state.name = name_value
                                    for event in emit_tool_item_added(state):
                                        yield event
                                    arguments_payload = function_obj.get("arguments")
                                    obfuscation_hint = None
                                    arguments_delta = ""
                                    if isinstance(arguments_payload, str):
                                        arguments_delta = arguments_payload
                                    elif isinstance(arguments_payload, dict):
                                        maybe_delta = arguments_payload.get("delta")
                                        if isinstance(maybe_delta, str):
                                            arguments_delta = maybe_delta
                                        obfuscation_hint = arguments_payload.get(
                                            "obfuscation"
                                        ) or arguments_payload.get("obfuscated")
                                    if arguments_delta:
                                        state.add_arguments_part(arguments_delta)
                                        sequence_counter += 1
                                        event_sequence = sequence_counter
                                        yield (
                                            delta_event_cls(
                                                type="response.function_call_arguments.delta",
                                                sequence_number=event_sequence,
                                                item_id=state.item_id
                                                or f"call_{state.index}",
                                                output_index=state.output_index,
                                                delta=arguments_delta,
                                            )
                                        )
                            for tool_call in tool_calls:
                                if not isinstance(tool_call, dict):
                                    continue
                                index_value = int(tool_call.get("index", 0))
                                state = get_tool_state(index_value)
                                if state.name:
                                    for event in emit_tool_item_added(state):
                                        yield event

                    usage_model: openai_models.ResponseUsage | None = None
                    if usage_obj is not None:
                        try:
                            if isinstance(usage_obj, openai_models.ResponseUsage):
                                usage_model = usage_obj
                            elif isinstance(usage_obj, dict):
                                usage_model = convert_usage(
                                    openai_models.CompletionUsage.model_validate(
                                        usage_obj
                                    )
                                )
                            else:
                                usage_model = convert_usage(usage_obj)
                        except Exception:
                            usage_model = None

                    if usage_model is not None:
                        latest_usage_model = usage_model
                        if all(reason is None for reason in finish_reasons):
                            sequence_counter += 1
                            yield openai_models.ResponseInProgressEvent(
                                type="response.in_progress",
                                sequence_number=sequence_counter,
                                response=make_response_object(
                                    status="in_progress",
                                    model=model or last_model,
                                    usage=usage_model,
                                    output=[],
                                    parallel_override=parallel_setting_initial,
                                ),
                            )

                    if any(reason == "tool_calls" for reason in finish_reasons):
                        for event in finalize_message():
                            yield event
                        for event in finalize_tool_calls():
                            yield event

            finally:
                register_request(None)
                register_request_tools(None)

            for event in finalize_reasoning():
                yield event

            for event in finalize_message():
                yield event

            for event in finalize_tool_calls():
                yield event

            if message_completed_entry is None and message_item_added:
                final_text = "".join(message_text_buffer)
                logprobs_value: Any
                if message_last_logprobs is None:
                    logprobs_value = []
                else:
                    logprobs_value = message_last_logprobs
                output_text_part = openai_models.OutputTextContent(
                    type="output_text",
                    text=final_text,
                    annotations=[],
                    logprobs=logprobs_value if logprobs_value != [] else [],
                )
                message_completed_entry = (
                    message_output_index or 0,
                    openai_models.MessageOutput(
                        type="message",
                        id=message_item_id,
                        status="completed",
                        role="assistant",
                        content=[output_text_part] if final_text else [],
                    ),
                )

            completed_entries: list[tuple[int, Any]] = []
            if reasoning_completed_entry is not None:
                completed_entries.append(reasoning_completed_entry)
            if message_completed_entry is not None:
                completed_entries.append(message_completed_entry)

            for idx, state in tool_call_states.items():
                accumulator_entry = get_accumulator_entry(idx)
                if state.final_arguments is None:
                    aggregated = ""
                    if accumulator_entry is not None:
                        aggregated = (
                            accumulator_entry.get("function", {}).get("arguments") or ""
                        )
                    if not aggregated:
                        aggregated = "".join(state.arguments_parts or [])
                    state.final_arguments = aggregated or ""
                if state.name is None and accumulator_entry is not None:
                    fn_name = accumulator_entry.get("function", {}).get("name")
                    if isinstance(fn_name, str) and fn_name:
                        state.name = fn_name
                if not state.item_id:
                    candidate_id = None
                    if accumulator_entry is not None:
                        candidate_id = accumulator_entry.get("id")
                    state.item_id = candidate_id or f"call_{state.index}"
                completed_entries.append(
                    (
                        state.output_index,
                        openai_models.FunctionCallOutput(
                            type="function_call",
                            id=state.item_id,
                            status="completed",
                            name=state.name,
                            call_id=state.call_id,
                            arguments=state.final_arguments or "",
                        ),
                    )
                )

            completed_entries.sort(key=lambda item: item[0])
            completed_outputs = [entry for _, entry in completed_entries]

            complete_tool_calls_payload = openai_accumulator.get_complete_tool_calls()
            parallel_tool_calls = len(tool_call_states) > 1
            parallel_final = parallel_tool_calls or parallel_setting_initial

            extra_fields: dict[str, Any] | None = None
            if complete_tool_calls_payload:
                extra_fields = {"tool_calls": complete_tool_calls_payload}

            response_completed = make_response_object(
                status="completed",
                model=last_model,
                usage=latest_usage_model,
                output=completed_outputs,
                parallel_override=parallel_final,
                reasoning_summary=reasoning_summary_payload,
                extra=extra_fields,
            )

            sequence_counter += 1
            yield openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=sequence_counter,
                response=response_completed,
            )

        return generator()


def convert__openai_chat_to_openai_responses__stream(
    stream: AsyncIterator[openai_models.ChatCompletionChunk | dict[str, Any]],
) -> AsyncGenerator[openai_models.StreamEventType, None]:
    """Convert OpenAI ChatCompletionChunk stream to Responses API events.

    Replays chat deltas as Responses events, including function-call output items
    and argument deltas so partial tool calls stream correctly.
    """
    adapter = OpenAIChatToResponsesStreamAdapter()
    return adapter.run(stream)


__all__ = [
    "OpenAIChatToResponsesStreamAdapter",
    "OpenAIResponsesToChatStreamAdapter",
    "convert__openai_chat_to_openai_responses__stream",
    "convert__openai_responses_to_openai_chat__stream",
]
