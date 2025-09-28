import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel, ValidationError

import ccproxy.core.logging
from ccproxy.llms.formatters.constants import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
)
from ccproxy.llms.formatters.context import (
    get_last_instructions,
    get_last_request,
    register_request,
)
from ccproxy.llms.formatters.utils import (
    anthropic_usage_snapshot,
    build_obfuscation_token,
)
from ccproxy.llms.models import anthropic as anthropic_models
from ccproxy.llms.models import openai as openai_models
from ccproxy.llms.streaming.accumulators import ClaudeAccumulator


logger = ccproxy.core.logging.get_logger(__name__)

FinishReason = Literal["stop", "length", "tool_calls"]


def _normalize_suffix(identifier: str) -> str:
    if "_" in identifier:
        return identifier.split("_", 1)[1]
    return identifier


def _ensure_identifier(prefix: str, existing: str | None = None) -> tuple[str, str]:
    if isinstance(existing, str) and existing.startswith(f"{prefix}_"):
        return existing, _normalize_suffix(existing)
    if isinstance(existing, str) and existing.startswith("resp_"):
        suffix = _normalize_suffix(existing)
        return f"{prefix}_{suffix}", suffix
    suffix = uuid.uuid4().hex
    return f"{prefix}_{suffix}", suffix


def convert__anthropic_usage_to_openai_completion__usage(
    usage: anthropic_models.Usage,
) -> openai_models.CompletionUsage:
    snapshot = anthropic_usage_snapshot(usage)

    cached_tokens = snapshot.cache_read_tokens or snapshot.cache_creation_tokens

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=cached_tokens, audio_tokens=0
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=0,
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


def convert__anthropic_usage_to_openai_responses__usage(
    usage: anthropic_models.Usage,
) -> openai_models.ResponseUsage:
    snapshot = anthropic_usage_snapshot(usage)

    cached_tokens = snapshot.cache_read_tokens or snapshot.cache_creation_tokens

    input_tokens_details = openai_models.InputTokensDetails(cached_tokens=cached_tokens)
    output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)

    return openai_models.ResponseUsage(
        input_tokens=snapshot.input_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=snapshot.output_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=snapshot.input_tokens + snapshot.output_tokens,
    )


# Error helpers


def convert__anthropic_to_openai__error(error: BaseModel) -> BaseModel:
    """Convert an Anthropic error payload to the OpenAI envelope."""
    from ccproxy.llms.models.anthropic import ErrorResponse as AnthropicErrorResponse
    from ccproxy.llms.models.openai import ErrorDetail
    from ccproxy.llms.models.openai import ErrorResponse as OpenAIErrorResponse

    if isinstance(error, AnthropicErrorResponse):
        anthropic_error = error.error
        error_message = anthropic_error.message
        anthropic_error_type = "api_error"
        if hasattr(anthropic_error, "type"):
            anthropic_error_type = anthropic_error.type

        openai_error_type = ANTHROPIC_TO_OPENAI_ERROR_TYPE.get(
            anthropic_error_type, "api_error"
        )

        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type=openai_error_type,
                code=None,
                param=None,
            )
        )

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type="api_error",
                code=None,
                param=None,
            )
        )

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        error_dict = error.model_dump()
        if isinstance(error_dict, dict):
            error_message = error_dict.get("message", str(error_dict))

    return OpenAIErrorResponse(
        error=ErrorDetail(
            message=error_message,
            type="api_error",
            code=None,
            param=None,
        )
    )


async def convert__anthropic_message_to_openai_responses__stream(
    stream: AsyncIterator[anthropic_models.MessageStreamEvent],
) -> AsyncGenerator[openai_models.StreamEventType, None]:
    """Convert Anthropic MessageStreamEvents into OpenAI Responses stream events."""

    accumulator = ClaudeAccumulator()
    sequence_counter = -1
    model_id = ""
    response_id = ""
    id_suffix: str | None = None
    message_item_id = ""
    message_output_index: int | None = None
    next_output_index = 0
    content_index = 0
    message_item_added = False
    message_content_part_added = False
    text_buffer: list[str] = []
    message_last_logprobs: Any | None = None
    message_text_done_emitted = False
    message_part_done_emitted = False
    message_item_done_emitted = False
    message_completed_entry: tuple[int, openai_models.MessageOutput] | None = None
    latest_usage_model: openai_models.ResponseUsage | None = None
    final_stop_reason: str | None = None
    stream_completed = False

    reasoning_item_id = ""
    reasoning_output_index: int | None = None
    reasoning_item_added = False
    reasoning_output_done = False
    reasoning_summary_indices: dict[str, int] = {}
    reasoning_summary_added: set[int] = set()
    reasoning_summary_text_fragments: dict[int, list[str]] = {}
    reasoning_summary_text_done: set[int] = set()
    reasoning_summary_part_done: set[int] = set()
    reasoning_completed_entry: tuple[int, openai_models.ReasoningOutput] | None = None
    next_reasoning_summary_index = 0
    reasoning_summary_signatures: dict[int, str | None] = {}
    created_at_value: int | None = None

    def make_obfuscation_token(
        kind: str,
        *,
        sequence: int,
        item_id: str | None = None,
        payload: str | None = None,
    ) -> str:
        base_identifier = item_id or id_suffix or response_id or "stream"
        seed = f"{kind}:{base_identifier}"
        return build_obfuscation_token(
            seed=seed,
            sequence=sequence,
            payload=payload or "",
        )

    instructions_text = get_last_instructions()
    if not instructions_text:
        try:
            from ccproxy.core.request_context import RequestContext

            ctx = RequestContext.get_current()
            if ctx is not None:
                instr = ctx.metadata.get("instructions")
                if isinstance(instr, str) and instr.strip():
                    instructions_text = instr.strip()
        except Exception:
            pass

    instructions_value = instructions_text or None

    envelope_base_kwargs: dict[str, Any] = {
        "id": "",
        "object": "response",
        "created_at": 0,
        "instructions": instructions_value,
    }
    reasoning_summary_payload: list[dict[str, Any]] | None = None

    last_request = get_last_request()
    anthropic_request: anthropic_models.CreateMessageRequest | None = None
    if isinstance(last_request, anthropic_models.CreateMessageRequest):
        anthropic_request = last_request
    elif isinstance(last_request, dict):
        try:
            anthropic_request = anthropic_models.CreateMessageRequest.model_validate(
                last_request
            )
        except ValidationError:
            anthropic_request = None

    base_parallel_tool_calls = True
    text_payload: dict[str, Any] | None = None

    if anthropic_request is not None:
        payload_data, _ = _build_responses_payload_from_anthropic_request(
            anthropic_request
        )
        base_parallel_tool_calls = bool(payload_data.get("parallel_tool_calls", True))
        envelope_base_kwargs["background"] = bool(payload_data.get("background", False))
        for key in (
            "max_output_tokens",
            "tool_choice",
            "tools",
            "service_tier",
            "temperature",
            "prompt_cache_key",
            "top_p",
            "metadata",
        ):
            if key in payload_data:
                envelope_base_kwargs[key] = payload_data[key]
        text_payload = payload_data.get("text")
    else:
        envelope_base_kwargs["background"] = False

    if text_payload is None:
        text_payload = {"format": {"type": "text"}}
    else:
        text_payload = dict(text_payload)
    text_payload.setdefault("verbosity", "low")
    envelope_base_kwargs["text"] = text_payload

    if "store" not in envelope_base_kwargs:
        envelope_base_kwargs["store"] = True

    if "temperature" not in envelope_base_kwargs:
        temp_value = None
        if anthropic_request is not None:
            temp_value = anthropic_request.temperature
        envelope_base_kwargs["temperature"] = (
            temp_value if temp_value is not None else 1.0
        )

    if "service_tier" not in envelope_base_kwargs:
        service_value = None
        if anthropic_request is not None:
            service_value = anthropic_request.service_tier
        envelope_base_kwargs["service_tier"] = service_value or "auto"

    if "top_p" not in envelope_base_kwargs:
        top_p_value = None
        if anthropic_request is not None:
            top_p_value = anthropic_request.top_p
        envelope_base_kwargs["top_p"] = top_p_value if top_p_value is not None else 1.0

    if "metadata" not in envelope_base_kwargs:
        envelope_base_kwargs["metadata"] = {}

    reasoning_effort = None
    if anthropic_request is not None:
        thinking_cfg = getattr(anthropic_request, "thinking", None)
        if getattr(thinking_cfg, "type", None) == "enabled":
            reasoning_effort = "medium"
    envelope_base_kwargs["reasoning"] = openai_models.Reasoning(
        effort=reasoning_effort,
        summary=None,
    )

    if "tool_choice" not in envelope_base_kwargs:
        envelope_base_kwargs["tool_choice"] = "auto"
    if "tools" not in envelope_base_kwargs:
        envelope_base_kwargs["tools"] = []

    parallel_setting_initial = bool(base_parallel_tool_calls)
    envelope_base_kwargs["parallel_tool_calls"] = parallel_setting_initial

    tool_states: dict[int, dict[str, Any]] = {}

    def ensure_message_output_item() -> list[openai_models.StreamEventType]:
        nonlocal message_item_added, message_output_index, next_output_index
        events: list[openai_models.StreamEventType] = []
        if message_output_index is None:
            message_output_index = next_output_index
            next_output_index += 1
        if not message_item_added:
            message_item_added = True
            nonlocal sequence_counter
            sequence_counter += 1
            events.append(
                openai_models.ResponseOutputItemAddedEvent(
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
            )
        return events

    def ensure_message_content_part() -> list[openai_models.StreamEventType]:
        events = ensure_message_output_item()
        nonlocal message_content_part_added, sequence_counter
        if not message_content_part_added and message_output_index is not None:
            message_content_part_added = True
            sequence_counter += 1
            events.append(
                openai_models.ResponseContentPartAddedEvent(
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
            )
        return events

    def emit_message_text_delta(
        text_delta: str,
        *,
        logprobs: Any | None = None,
        obfuscation: str | None = None,
    ) -> list[openai_models.StreamEventType]:
        if not isinstance(text_delta, str) or not text_delta:
            return []

        nonlocal sequence_counter, message_last_logprobs, message_item_done_emitted
        if message_item_done_emitted:
            return []

        events = ensure_message_content_part()
        sequence_counter += 1
        event_sequence = sequence_counter
        logprobs_value: Any = [] if logprobs is None else logprobs
        events.append(
            openai_models.ResponseOutputTextDeltaEvent(
                type="response.output_text.delta",
                sequence_number=event_sequence,
                item_id=message_item_id,
                output_index=message_output_index or 0,
                content_index=content_index,
                delta=text_delta,
                logprobs=logprobs_value,
            )
        )
        text_buffer.append(text_delta)
        message_last_logprobs = logprobs_value
        return events

    def _reasoning_key(signature: str | None) -> str:
        if isinstance(signature, str) and signature.strip():
            return signature.strip()
        return "__default__"

    def get_reasoning_summary_index(signature: str | None) -> int:
        nonlocal next_reasoning_summary_index
        key = _reasoning_key(signature)
        existing = reasoning_summary_indices.get(key)
        if existing is not None:
            return existing
        reasoning_summary_indices[key] = next_reasoning_summary_index
        reasoning_summary_signatures[next_reasoning_summary_index] = signature
        next_reasoning_summary_index += 1
        return reasoning_summary_indices[key]

    def ensure_reasoning_output_item() -> (
        openai_models.ResponseOutputItemAddedEvent | None
    ):
        nonlocal reasoning_item_added, reasoning_output_index
        nonlocal sequence_counter, next_output_index
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

    def emit_reasoning_text_delta(
        text_delta: str,
        signature: str | None,
    ) -> list[openai_models.StreamEventType]:
        if not isinstance(text_delta, str) or not text_delta:
            return []

        events: list[openai_models.StreamEventType] = []
        output_event = ensure_reasoning_output_item()
        if output_event is not None:
            events.append(output_event)

        summary_index = get_reasoning_summary_index(signature)
        part_event = ensure_reasoning_summary_part(summary_index)
        if part_event is not None:
            events.append(part_event)

        fragments = reasoning_summary_text_fragments.setdefault(summary_index, [])
        fragments.append(text_delta)
        if summary_index not in reasoning_summary_signatures:
            reasoning_summary_signatures[summary_index] = signature

        nonlocal sequence_counter
        sequence_counter += 1
        event_sequence = sequence_counter
        events.append(
            openai_models.ReasoningSummaryTextDeltaEvent(
                type="response.reasoning_summary_text.delta",
                sequence_number=event_sequence,
                item_id=reasoning_item_id,
                output_index=reasoning_output_index or 0,
                summary_index=summary_index,
                delta=text_delta,
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

    def ensure_tool_state(block_index: int) -> dict[str, Any]:
        nonlocal next_output_index
        state = tool_states.get(block_index)
        if state is None:
            state = {
                "block_index": block_index,
                "output_index": next_output_index,
                "item_id": None,
                "name": None,
                "call_id": None,
                "arguments_parts": [],
                "added_emitted": False,
                "arguments_done_emitted": False,
                "item_done_emitted": False,
            }
            tool_states[block_index] = state
            next_output_index += 1
        return state

    def emit_tool_item_added(
        block_index: int, state: dict[str, Any]
    ) -> list[openai_models.StreamEventType]:
        events: list[openai_models.StreamEventType] = []
        if state.get("added_emitted"):
            return events

        tool_entry = accumulator.get_tool_entry(block_index)
        if tool_entry:
            state.setdefault(
                "name",
                tool_entry.get("function", {}).get("name") or tool_entry.get("name"),
            )
            state.setdefault("call_id", tool_entry.get("id"))

        item_id = state.get("item_id") or state.get("call_id")
        if not item_id:
            item_id = f"call_{block_index}"
        state["item_id"] = item_id

        name = state.get("name") or "function"

        nonlocal sequence_counter
        sequence_counter += 1
        events.append(
            openai_models.ResponseOutputItemAddedEvent(
                type="response.output_item.added",
                sequence_number=sequence_counter,
                output_index=state["output_index"],
                item=openai_models.OutputItem(
                    id=str(item_id),
                    type="function_call",
                    status="in_progress",
                    name=str(name),
                    arguments="",
                    call_id=state.get("call_id"),
                ),
            )
        )
        state["added_emitted"] = True
        return events

    def emit_tool_arguments_delta(
        state: dict[str, Any], delta_text: str
    ) -> openai_models.StreamEventType:
        nonlocal sequence_counter
        sequence_counter += 1
        event_sequence = sequence_counter
        state.setdefault("arguments_parts", []).append(delta_text)
        item_identifier = str(state.get("item_id") or f"call_{state['block_index']}")
        return openai_models.ResponseFunctionCallArgumentsDeltaEvent(
            type="response.function_call_arguments.delta",
            sequence_number=event_sequence,
            item_id=item_identifier,
            output_index=state["output_index"],
            delta=delta_text,
        )

    def emit_tool_finalize(
        block_index: int, state: dict[str, Any]
    ) -> list[openai_models.StreamEventType]:
        events: list[openai_models.StreamEventType] = []
        tool_entry = accumulator.get_tool_entry(block_index)

        if tool_entry:
            state.setdefault(
                "name",
                tool_entry.get("function", {}).get("name") or tool_entry.get("name"),
            )
            state.setdefault("call_id", tool_entry.get("id"))
            state.setdefault("item_id", tool_entry.get("id"))

        item_id = state.get("item_id") or state.get("call_id") or f"call_{block_index}"
        state["item_id"] = item_id
        name = state.get("name") or "function"

        args_str = "".join(state.get("arguments_parts", []))
        if not args_str and tool_entry:
            try:
                args_str = json.dumps(tool_entry.get("input", {}), ensure_ascii=False)
            except Exception:
                args_str = json.dumps(tool_entry.get("input", {}))

        nonlocal sequence_counter
        if not state.get("added_emitted"):
            events.extend(emit_tool_item_added(block_index, state))

        if not state.get("arguments_done_emitted"):
            sequence_counter += 1
            events.append(
                openai_models.ResponseFunctionCallArgumentsDoneEvent(
                    type="response.function_call_arguments.done",
                    sequence_number=sequence_counter,
                    item_id=str(item_id),
                    output_index=state["output_index"],
                    arguments=args_str,
                )
            )
            state["arguments_done_emitted"] = True

        if not state.get("item_done_emitted"):
            sequence_counter += 1
            events.append(
                openai_models.ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=sequence_counter,
                    output_index=state["output_index"],
                    item=openai_models.OutputItem(
                        id=str(item_id),
                        type="function_call",
                        status="completed",
                        name=str(name),
                        arguments=args_str,
                        call_id=state.get("call_id"),
                    ),
                )
            )
            state["item_done_emitted"] = True
            state["final_arguments"] = args_str

        return events

    def finalize_message() -> list[openai_models.StreamEventType]:
        nonlocal sequence_counter
        nonlocal message_text_done_emitted, message_part_done_emitted
        nonlocal message_item_done_emitted, message_completed_entry
        nonlocal message_last_logprobs
        nonlocal accumulator

        if not message_item_added or message_output_index is None:
            return []

        events: list[openai_models.StreamEventType] = []
        final_text = "".join(text_buffer)
        logprobs_value: Any
        if message_last_logprobs is None:
            logprobs_value = []
        else:
            logprobs_value = message_last_logprobs

        primary_text_part: openai_models.OutputTextContent | None = None
        tool_and_aux_blocks: list[Any] = []

        if accumulator.content_blocks:
            sorted_blocks = sorted(
                accumulator.content_blocks, key=lambda block: block.get("index", 0)
            )
            for block in sorted_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    text_value = block.get("text", "")
                    part = openai_models.OutputTextContent(
                        type="output_text",
                        text=text_value,
                        annotations=[],
                        logprobs=logprobs_value if text_value else [],
                    )
                    if primary_text_part is None and text_value:
                        primary_text_part = part
                    tool_and_aux_blocks.append(part)
                else:
                    block_payload = {k: v for k, v in block.items() if k != "index"}
                    if block_payload.get("type") == "tool_use":
                        tool_input = block_payload.get("input")
                        if tool_input is not None:
                            block_payload.setdefault("arguments", tool_input)
                    tool_and_aux_blocks.append(block_payload)

        if primary_text_part is None and final_text:
            primary_text_part = openai_models.OutputTextContent(
                type="output_text",
                text=final_text,
                annotations=[],
                logprobs=logprobs_value if final_text else [],
            )
            tool_and_aux_blocks.insert(0, primary_text_part)

        if message_content_part_added and not message_text_done_emitted:
            sequence_counter += 1
            event_sequence = sequence_counter
            events.append(
                openai_models.ResponseOutputTextDoneEvent(
                    type="response.output_text.done",
                    sequence_number=event_sequence,
                    item_id=message_item_id,
                    output_index=message_output_index,
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
                    output_index=message_output_index,
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
            if primary_text_part is None:
                primary_text_part = openai_models.OutputTextContent(
                    type="output_text",
                    text=final_text,
                    annotations=[],
                    logprobs=logprobs_value if logprobs_value != [] else [],
                )
                tool_and_aux_blocks.insert(0, primary_text_part)
            message_output = openai_models.MessageOutput(
                type="message",
                id=message_item_id,
                status="completed",
                role="assistant",
                content=tool_and_aux_blocks,
            )
            message_completed_entry = (message_output_index, message_output)
            events.append(
                openai_models.ResponseOutputItemDoneEvent(
                    type="response.output_item.done",
                    sequence_number=event_sequence,
                    output_index=message_output_index,
                    item=openai_models.OutputItem(
                        id=message_item_id,
                        type="message",
                        role="assistant",
                        status="completed",
                        content=[
                            part.model_dump() if hasattr(part, "model_dump") else part
                            for part in tool_and_aux_blocks
                        ],
                        text=final_text or None,
                    ),
                )
            )
            message_item_done_emitted = True
        else:
            if primary_text_part is None and final_text:
                primary_text_part = openai_models.OutputTextContent(
                    type="output_text",
                    text=final_text,
                    annotations=[],
                    logprobs=logprobs_value if logprobs_value != [] else [],
                )
                tool_and_aux_blocks.insert(0, primary_text_part)
            message_completed_entry = (
                message_output_index,
                openai_models.MessageOutput(
                    type="message",
                    id=message_item_id,
                    status="completed",
                    role="assistant",
                    content=tool_and_aux_blocks,
                ),
            )

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
        async for raw_event in stream:
            event_type, event_payload = _normalize_anthropic_stream_event(raw_event)
            if not event_type:
                continue

            accumulator.accumulate(event_type, event_payload)

            if event_type == "ping":
                continue

            if event_type == "error":
                continue

            if event_type == "message_start":
                message = (
                    event_payload.get("message", {})
                    if isinstance(event_payload, dict)
                    else {}
                )
                model_id = str(message.get("model", ""))
                response_id, id_suffix = _ensure_identifier("resp", message.get("id"))
                envelope_base_kwargs["id"] = response_id
                envelope_base_kwargs.setdefault("object", "response")
                if model_id:
                    envelope_base_kwargs["model"] = model_id
                if not message_item_id:
                    message_item_id = f"msg_{id_suffix}"
                if not reasoning_item_id:
                    reasoning_item_id = f"rs_{id_suffix}"

                created_at_value = (
                    message.get("created_at")
                    or message.get("created")
                    or int(time.time())
                )
                envelope_base_kwargs["created_at"] = int(created_at_value)

                sequence_counter += 1
                yield openai_models.ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=sequence_counter,
                    response=make_response_object(
                        status="in_progress",
                        model=model_id,
                        usage=None,
                        output=[],
                        parallel_override=parallel_setting_initial,
                    ),
                )
                sequence_counter += 1
                yield openai_models.ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=sequence_counter,
                    response=make_response_object(
                        status="in_progress",
                        model=model_id,
                        usage=latest_usage_model,
                        output=[],
                        parallel_override=parallel_setting_initial,
                    ),
                )
                continue

            if event_type == "content_block_start":
                block_index = int(event_payload.get("index", 0))
                content_block = (
                    event_payload.get("content_block", {})
                    if isinstance(event_payload, dict)
                    else {}
                )
                if (
                    isinstance(content_block, dict)
                    and content_block.get("type") == "tool_use"
                ):
                    state = ensure_tool_state(block_index)
                    state["name"] = content_block.get("name") or state.get("name")
                    state["call_id"] = content_block.get("id") or state.get("call_id")
                    state["item_id"] = state.get("item_id") or content_block.get("id")
                    for event in finalize_message():
                        yield event
                    for event in emit_tool_item_added(block_index, state):
                        yield event
                continue

            if event_type == "content_block_delta":
                block_index = int(event_payload.get("index", 0))
                block_info = accumulator.get_block_info(block_index)
                if not block_info:
                    continue
                _, block_meta = block_info
                delta_payload = event_payload.get("delta")

                block_type = block_meta.get("type")

                if block_type == "thinking" and isinstance(delta_payload, dict):
                    thinking_text = delta_payload.get("thinking")
                    if isinstance(thinking_text, str) and thinking_text:
                        signature = block_meta.get("signature")
                        for event in emit_reasoning_text_delta(
                            thinking_text, signature
                        ):
                            yield event
                    continue

                if block_type == "text" and isinstance(delta_payload, dict):
                    text_delta = delta_payload.get("text")
                    if isinstance(text_delta, str) and text_delta:
                        for event in emit_message_text_delta(
                            text_delta,
                            logprobs=delta_payload.get("logprobs"),
                            obfuscation=delta_payload.get("obfuscation")
                            or delta_payload.get("obfuscated"),
                        ):
                            yield event
                    continue

                if block_type == "tool_use" and isinstance(delta_payload, dict):
                    partial = delta_payload.get("partial_json") or ""
                    if partial:
                        state = ensure_tool_state(block_index)
                        for event in finalize_message():
                            yield event
                        for event in emit_tool_item_added(block_index, state):
                            yield event
                        yield emit_tool_arguments_delta(
                            state,
                            str(partial),
                        )
                continue

            if event_type == "content_block_stop":
                block_index = int(event_payload.get("index", 0))
                block_info = accumulator.get_block_info(block_index)
                if block_info and block_info[1].get("type") == "tool_use":
                    state = ensure_tool_state(block_index)
                    for event in emit_tool_finalize(block_index, state):
                        yield event
                continue

            if event_type == "message_delta":
                delta_payload = (
                    event_payload.get("delta", {})
                    if isinstance(event_payload, dict)
                    else {}
                )
                stop_reason = (
                    delta_payload.get("stop_reason")
                    if isinstance(delta_payload, dict)
                    else None
                )
                if isinstance(stop_reason, str):
                    final_stop_reason = stop_reason

                usage_payload = (
                    event_payload.get("usage")
                    if isinstance(event_payload, dict)
                    else None
                )
                usage_model: anthropic_models.Usage | None = None
                if usage_payload:
                    try:
                        usage_model = anthropic_models.Usage.model_validate(
                            usage_payload
                        )
                    except ValidationError:
                        usage_model = anthropic_models.Usage(
                            input_tokens=usage_payload.get("input_tokens", 0),
                            output_tokens=usage_payload.get("output_tokens", 0),
                        )
                elif hasattr(raw_event, "usage") and raw_event.usage is not None:
                    usage_model = raw_event.usage

                if usage_model is not None:
                    latest_usage_model = (
                        convert__anthropic_usage_to_openai_responses__usage(usage_model)
                    )

                sequence_counter += 1
                yield openai_models.ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=sequence_counter,
                    response=make_response_object(
                        status="in_progress",
                        model=model_id,
                        usage=latest_usage_model,
                        output=[],
                        parallel_override=parallel_setting_initial,
                    ),
                )
                continue

            if event_type == "message_stop":
                for event in finalize_reasoning():
                    yield event

                for event in finalize_message():
                    yield event

                for index, state in list(tool_states.items()):
                    for event in emit_tool_finalize(index, state):
                        yield event

                first_completed_entries: list[tuple[int, Any]] = []
                if reasoning_completed_entry is not None:
                    first_completed_entries.append(reasoning_completed_entry)
                if message_completed_entry is not None:
                    first_completed_entries.append(message_completed_entry)

                for index, state in sorted(tool_states.items()):
                    tool_entry = accumulator.get_tool_entry(index)
                    if state.get("name") is None and tool_entry is not None:
                        state["name"] = tool_entry.get("name") or tool_entry.get(
                            "function", {}
                        ).get("name")
                    if state.get("call_id") is None and tool_entry is not None:
                        state["call_id"] = tool_entry.get("id")
                    if not state.get("item_id"):
                        state["item_id"] = (
                            state.get("call_id") or f"call_{state['block_index']}"
                        )

                    final_args = state.get("final_arguments")
                    if final_args is None:
                        combined = "".join(state.get("arguments_parts", []))
                        if not combined and tool_entry is not None:
                            input_payload = tool_entry.get("input", {}) or {}
                            try:
                                combined = json.dumps(input_payload, ensure_ascii=False)
                            except Exception:
                                combined = json.dumps(input_payload)
                        final_args = combined or ""
                    state["final_arguments"] = final_args

                    first_completed_entries.append(
                        (
                            state["output_index"],
                            openai_models.FunctionCallOutput(
                                type="function_call",
                                id=state["item_id"],
                                status="completed",
                                name=state.get("name"),
                                call_id=state.get("call_id"),
                                arguments=final_args,
                            ),
                        )
                    )

                first_completed_entries.sort(key=lambda item: item[0])
                completed_outputs = [entry for _, entry in first_completed_entries]

                complete_tool_calls_payload = accumulator.get_complete_tool_calls()
                parallel_final = parallel_setting_initial or len(tool_states) > 1

                extra_fields: dict[str, Any] | None = None
                if complete_tool_calls_payload:
                    extra_fields = {"tool_calls": complete_tool_calls_payload}

                status_value = "completed"
                if final_stop_reason == "max_tokens":
                    status_value = "incomplete"

                completed_response = make_response_object(
                    status=status_value,
                    model=model_id,
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
                    response=completed_response,
                )
                stream_completed = True
                break

        if not stream_completed:
            for event in finalize_reasoning():
                yield event

            for event in finalize_message():
                yield event

            for index, state in list(tool_states.items()):
                for event in emit_tool_finalize(index, state):
                    yield event

            if (
                message_completed_entry is None
                and message_item_added
                and message_output_index is not None
            ):
                final_text = "".join(text_buffer)
                logprobs_value: Any
                if message_last_logprobs is None:
                    logprobs_value = []
                else:
                    logprobs_value = message_last_logprobs
                content_blocks: list[Any] = []
                if accumulator.content_blocks:
                    sorted_blocks = sorted(
                        accumulator.content_blocks,
                        key=lambda block: block.get("index", 0),
                    )
                    for block in sorted_blocks:
                        block_type = block.get("type")
                        if block_type == "text":
                            text_value = block.get("text", "")
                            content_blocks.append(
                                openai_models.OutputTextContent(
                                    type="output_text",
                                    text=text_value,
                                    annotations=[],
                                    logprobs=logprobs_value if text_value else [],
                                )
                            )
                        else:
                            payload = {k: v for k, v in block.items() if k != "index"}
                            if payload.get("type") == "tool_use":
                                tool_input = payload.get("input")
                                if tool_input is not None:
                                    payload.setdefault("arguments", tool_input)
                            content_blocks.append(payload)
                else:
                    if final_text:
                        content_blocks.append(
                            openai_models.OutputTextContent(
                                type="output_text",
                                text=final_text,
                                annotations=[],
                                logprobs=logprobs_value if logprobs_value != [] else [],
                            )
                        )

                message_completed_entry = (
                    message_output_index,
                    openai_models.MessageOutput(
                        type="message",
                        id=message_item_id,
                        status="completed",
                        role="assistant",
                        content=content_blocks,
                    ),
                )

            final_completed_entries: list[tuple[int, Any]] = []
            if reasoning_completed_entry is not None:
                final_completed_entries.append(reasoning_completed_entry)
            if message_completed_entry is not None:
                final_completed_entries.append(message_completed_entry)

            for index, state in sorted(tool_states.items()):
                tool_entry = accumulator.get_tool_entry(index)
                if state.get("name") is None and tool_entry is not None:
                    state["name"] = tool_entry.get("name") or tool_entry.get(
                        "function", {}
                    ).get("name")
                if state.get("call_id") is None and tool_entry is not None:
                    state["call_id"] = tool_entry.get("id")
                if not state.get("item_id"):
                    state["item_id"] = (
                        state.get("call_id") or f"call_{state['block_index']}"
                    )
                final_args = state.get("final_arguments")
                if final_args is None:
                    combined = "".join(state.get("arguments_parts", []))
                    if not combined and tool_entry is not None:
                        input_payload = tool_entry.get("input", {}) or {}
                        try:
                            combined = json.dumps(input_payload, ensure_ascii=False)
                        except Exception:
                            combined = json.dumps(input_payload)
                    final_args = combined or ""
                state["final_arguments"] = final_args
                final_completed_entries.append(
                    (
                        state["output_index"],
                        openai_models.FunctionCallOutput(
                            type="function_call",
                            id=state["item_id"],
                            status="completed",
                            name=state.get("name"),
                            call_id=state.get("call_id"),
                            arguments=final_args,
                        ),
                    )
                )

            final_completed_entries.sort(key=lambda item: item[0])
            completed_outputs = [entry for _, entry in final_completed_entries]

            complete_tool_calls_payload = accumulator.get_complete_tool_calls()
            parallel_final = parallel_setting_initial or len(tool_states) > 1

            final_extra_fields: dict[str, Any] | None = None
            if complete_tool_calls_payload:
                final_extra_fields = {"tool_calls": complete_tool_calls_payload}

            fallback_response = make_response_object(
                status="completed",
                model=model_id,
                usage=latest_usage_model,
                output=completed_outputs,
                parallel_override=parallel_final,
                reasoning_summary=reasoning_summary_payload,
                extra=final_extra_fields,
            )

            sequence_counter += 1
            yield openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=sequence_counter,
                response=fallback_response,
            )

    finally:
        register_request(None)


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

    last_user_text: str | None = None
    for msg in reversed(request.messages):
        if msg.role != "user":
            continue
        if isinstance(msg.content, str):
            last_user_text = msg.content
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
                last_user_text = " ".join(texts)
        break

    if last_user_text:
        payload_data["input"] = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": last_user_text},
                ],
            }
        ]
    else:
        payload_data["input"] = []

    if request.tools:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if isinstance(tool, anthropic_models.Tool):
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

    payload_data.setdefault("background", False)

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


def _normalize_anthropic_stream_event(
    event: Any,
) -> tuple[str | None, dict[str, Any]]:
    """Return a (type, payload) tuple for mixed dict/model stream events."""

    if isinstance(event, dict):
        event_type = event.get("type") or event.get("event")
        return (cast(str | None, event_type), event)

    event_type = getattr(event, "type", None)
    if event_type is None:
        return None, {}

    if hasattr(event, "model_dump"):
        payload = cast(dict[str, Any], event.model_dump(mode="json"))
    elif hasattr(event, "dict"):
        payload = cast(dict[str, Any], event.dict())
    else:
        payload = {}

    return cast(str | None, event_type), payload


def _anthropic_delta_to_text(
    accumulator: ClaudeAccumulator,
    block_index: int,
    delta: dict[str, Any] | None,
) -> str | None:
    if not isinstance(delta, dict):
        return None

    block_info = accumulator.get_block_info(block_index)
    block_meta = block_info[1] if block_info else {}
    block_type = block_meta.get("type")

    if block_type == "thinking":
        thinking_text = delta.get("thinking")
        if not isinstance(thinking_text, str) or not thinking_text:
            return None
        signature = block_meta.get("signature")
        if isinstance(signature, str) and signature:
            return f'<thinking signature="{signature}">{thinking_text}</thinking>'
        return f"<thinking>{thinking_text}</thinking>"

    text_val = delta.get("text")
    if isinstance(text_val, str) and text_val:
        return text_val

    return None


def _build_openai_tool_call(
    accumulator: ClaudeAccumulator,
    block_index: int,
) -> openai_models.ToolCall | None:
    for tool_call in accumulator.get_complete_tool_calls():
        if tool_call.get("index") != block_index:
            continue

        function_payload = (
            tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        )
        name = function_payload.get("name") or tool_call.get("name") or "function"
        arguments = function_payload.get("arguments")
        if not isinstance(arguments, str) or not arguments:
            try:
                arguments = json.dumps(tool_call.get("input", {}), ensure_ascii=False)
            except Exception:
                arguments = json.dumps(tool_call.get("input", {}))

        tool_id = tool_call.get("id") or f"call_{block_index}"

        return openai_models.ToolCall(
            id=str(tool_id),
            function=openai_models.FunctionCall(
                name=str(name),
                arguments=str(arguments),
            ),
        )

    return None


def convert__anthropic_message_to_openai_chat__stream(
    stream: AsyncIterator[anthropic_models.MessageStreamEvent],
) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
    """Convert Anthropic stream to OpenAI stream using ClaudeAccumulator."""

    async def generator() -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        accumulator = ClaudeAccumulator()
        model_id = ""
        finish_reason: FinishReason = "stop"
        usage_prompt = 0
        usage_completion = 0
        message_started = False
        emitted_tool_indices: set[int] = set()

        async for raw_event in stream:
            event_type, event_payload = _normalize_anthropic_stream_event(raw_event)
            if not event_type:
                continue

            accumulator.accumulate(event_type, event_payload)

            if event_type == "ping":
                continue

            if event_type == "error":
                # Error events are handled elsewhere by callers.
                continue

            if event_type == "message_start":
                message_data = (
                    event_payload.get("message", {})
                    if isinstance(event_payload, dict)
                    else {}
                )
                model_id = str(message_data.get("model", ""))
                message_started = True
                yield openai_models.ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
                    created=0,
                    model=model_id,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(
                                role="assistant", content=""
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                continue

            if not message_started:
                continue

            if event_type == "content_block_delta":
                block_index = int(event_payload.get("index", 0))
                text_delta = _anthropic_delta_to_text(
                    accumulator,
                    block_index,
                    cast(dict[str, Any] | None, event_payload.get("delta")),
                )
                if text_delta:
                    yield openai_models.ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=text_delta
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
                continue

            if event_type == "content_block_stop":
                block_index = int(event_payload.get("index", 0))
                block_info = accumulator.get_block_info(block_index)
                if not block_info:
                    continue
                _, block_meta = block_info
                if block_meta.get("type") != "tool_use":
                    continue
                if block_index in emitted_tool_indices:
                    continue
                tool_call = _build_openai_tool_call(accumulator, block_index)
                if tool_call is None:
                    continue
                emitted_tool_indices.add(block_index)
                yield openai_models.ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
                    created=0,
                    model=model_id,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(
                                role="assistant", tool_calls=[tool_call]
                            ),
                            finish_reason=None,
                        )
                    ],
                )
                continue

            if event_type == "message_delta":
                delta_payload = (
                    event_payload.get("delta", {})
                    if isinstance(event_payload, dict)
                    else {}
                )
                stop_reason = (
                    delta_payload.get("stop_reason")
                    if isinstance(delta_payload, dict)
                    else None
                )
                if isinstance(stop_reason, str):
                    finish_reason = cast(
                        FinishReason,
                        ANTHROPIC_TO_OPENAI_FINISH_REASON.get(stop_reason, "stop"),
                    )

                usage_payload = (
                    event_payload.get("usage")
                    if isinstance(event_payload, dict)
                    else None
                )
                if usage_payload:
                    snapshot = anthropic_usage_snapshot(usage_payload)
                    usage_prompt = snapshot.input_tokens
                    usage_completion = snapshot.output_tokens
                elif hasattr(raw_event, "usage") and raw_event.usage is not None:
                    snapshot = anthropic_usage_snapshot(raw_event.usage)
                    usage_prompt = snapshot.input_tokens
                    usage_completion = snapshot.output_tokens
                continue

            if event_type == "message_stop":
                usage = None
                if usage_prompt or usage_completion:
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=usage_prompt,
                        completion_tokens=usage_completion,
                        total_tokens=usage_prompt + usage_completion,
                    )

                yield openai_models.ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
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
                break

        else:
            if message_started:
                yield openai_models.ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
                    created=0,
                    model=model_id,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                )

    return generator()


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
            signature = getattr(block, "signature", None)
            sig_attr = (
                f' signature="{signature}"'
                if isinstance(signature, str) and signature
                else ""
            )
            text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
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
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": args_str,
                                },
                            }
                        )
                elif block_type == "text":
                    # Type guard for TextBlock
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
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
                                    "tool_call_id": block.tool_use_id,
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
                    if (
                        btype == "text"
                        and hasattr(block, "text")
                        and isinstance(getattr(block, "text", None), str)
                    ):
                        text_accum.append(block.text or "")
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
            if isinstance(tool, anthropic_models.Tool):
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


def convert__anthropic_message_to_openai_chat__response(
    response: anthropic_models.MessageResponse,
) -> openai_models.ChatCompletionResponse:
    """Convert Anthropic MessageResponse to an OpenAI ChatCompletionResponse."""
    content_blocks = response.content
    parts: list[str] = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        elif btype == "thinking":
            thinking = getattr(block, "thinking", None)
            signature = getattr(block, "signature", None)
            if isinstance(thinking, str):
                sig_attr = (
                    f' signature="{signature}"'
                    if isinstance(signature, str) and signature
                    else ""
                )
                parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")

    content_text = "".join(parts)

    stop_reason = response.stop_reason
    finish_reason = ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
        stop_reason or "end_turn", "stop"
    )

    usage_model = convert__anthropic_usage_to_openai_completion__usage(response.usage)

    payload = {
        "id": response.id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": finish_reason,
            }
        ],
        "created": int(time.time()),
        "model": response.model,
        "object": "chat.completion",
        "usage": usage_model.model_dump(),
    }

    return openai_models.ChatCompletionResponse.model_validate(payload)
