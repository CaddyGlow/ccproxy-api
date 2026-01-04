"""Shared streaming helpers for formatter adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ccproxy.llms.formatters.utils import build_obfuscation_token
from ccproxy.llms.models import anthropic as anthropic_models

from .thinking import ThinkingSegment


@dataclass(slots=True)
class ReasoningPartState:
    """Mutable reasoning buffer for a specific summary segment."""

    buffer: list[str] = field(default_factory=list)
    signature: str | None = None
    open: bool = False


class ReasoningBuffer:
    """Utility to manage reasoning text buffers keyed by item/summary ids."""

    def __init__(self) -> None:
        self._states: dict[str, dict[Any, ReasoningPartState]] = {}

    def ensure_part(self, item_id: str, summary_index: Any) -> ReasoningPartState:
        item_states = self._states.setdefault(item_id, {})
        part_state = item_states.get(summary_index)
        if part_state is None:
            part_state = ReasoningPartState()
            item_states[summary_index] = part_state
        return part_state

    def set_signature(
        self, item_id: str, summary_index: Any, signature: str | None
    ) -> None:
        if not signature:
            return
        part_state = self.ensure_part(item_id, summary_index)
        part_state.signature = signature

    def reset_buffer(self, item_id: str, summary_index: Any) -> None:
        part_state = self.ensure_part(item_id, summary_index)
        part_state.buffer.clear()

    def open_part(
        self, item_id: str, summary_index: Any, signature: str | None = None
    ) -> ReasoningPartState:
        part_state = self.ensure_part(item_id, summary_index)
        if signature:
            part_state.signature = signature
        part_state.buffer.clear()
        part_state.open = True
        return part_state

    def close_part(self, item_id: str, summary_index: Any) -> None:
        part_state = self.ensure_part(item_id, summary_index)
        part_state.open = False

    def is_open(self, item_id: str, summary_index: Any) -> bool:
        return self.ensure_part(item_id, summary_index).open

    def append_text(self, item_id: str, summary_index: Any, text: str | None) -> None:
        if not isinstance(text, str) or not text:
            return
        part_state = self.ensure_part(item_id, summary_index)
        part_state.buffer.append(text)

    def emit(
        self, item_id: str, summary_index: Any, final_text: str | None = None
    ) -> list[str]:
        part_state = self.ensure_part(item_id, summary_index)
        text = (
            final_text
            if isinstance(final_text, str) and final_text
            else "".join(part_state.buffer)
        )
        part_state.buffer.clear()
        part_state.open = False
        if not text:
            return []
        segment = ThinkingSegment(thinking=text, signature=part_state.signature)
        xml = segment.to_xml()
        closing = "</thinking>"
        body = xml[: -len(closing)] if xml.endswith(closing) else xml
        return [body, closing]


@dataclass(slots=True)
class ToolCallState:
    """Mutable state for a single streaming tool call."""

    id: str
    index: int
    call_id: str | None = None
    item_id: str | None = None
    name: str | None = None
    arguments: str = ""
    arguments_parts: list[str] = field(default_factory=list)
    output_index: int = -1
    emitted: bool = False
    initial_emitted: bool = False
    name_emitted: bool = False
    arguments_emitted: bool = False
    arguments_done_emitted: bool = False
    item_done_emitted: bool = False
    added_emitted: bool = False
    completed: bool = False
    final_arguments: str | None = None

    def append_arguments(self, segment: str) -> None:
        if segment:
            self.arguments += segment

    def add_arguments_part(self, segment: str) -> None:
        if segment:
            self.arguments_parts.append(segment)


class ToolCallTracker:
    """Registry tracking streaming tool calls by item identifier."""

    def __init__(self) -> None:
        self._states: dict[str, ToolCallState] = {}
        self._order: list[str] = []

    def ensure(self, item_id: str) -> ToolCallState:
        state = self._states.get(item_id)
        if state is None:
            state = ToolCallState(
                id=item_id,
                index=len(self._order),
            )
            state.output_index = len(self._order)
            self._states[item_id] = state
            self._order.append(item_id)
        return state

    def values(self) -> list[ToolCallState]:
        return [self._states[item_id] for item_id in self._order]

    def any_completed(self) -> bool:
        return any(state.completed for state in self._states.values())

    def __len__(self) -> int:  # noqa: D401
        return len(self._states)


class IndexedToolCallTracker:
    """Registry tracking streaming tool calls keyed by integer index."""

    def __init__(self) -> None:
        self._states: dict[int, ToolCallState] = {}

    def ensure(self, index: int) -> ToolCallState:
        state = self._states.get(index)
        if state is None:
            state = ToolCallState(id=f"call_{index}", index=index)
            self._states[index] = state
        return state

    def items(self) -> list[tuple[int, ToolCallState]]:
        return [(idx, self._states[idx]) for idx in sorted(self._states)]

    def values(self) -> list[ToolCallState]:
        return [state for _, state in self.items()]

    def __contains__(self, index: int) -> bool:  # noqa: D401
        return index in self._states

    def __len__(self) -> int:  # noqa: D401
        return len(self._states)


class ObfuscationTokenFactory:
    """Utility for building deterministic obfuscation tokens."""

    def __init__(self, fallback_identifier: Callable[[], str]) -> None:
        self._fallback_identifier = fallback_identifier

    def make(
        self,
        kind: str,
        *,
        sequence: int,
        item_id: str | None = None,
        payload: str | None = None,
    ) -> str:
        base_identifier = item_id or self._fallback_identifier()
        return build_obfuscation_token(
            seed=f"{kind}:{base_identifier}",
            sequence=sequence,
            payload=payload or "",
        )


def build_anthropic_tool_use_block(
    state: ToolCallState,
    *,
    default_id: str | None = None,
    parser: Callable[[str], dict[str, Any]] | None = None,
) -> anthropic_models.ToolUseBlock:
    """Create an Anthropic ToolUseBlock from a tracked tool-call state."""

    tool_id = state.item_id or state.call_id or default_id or f"call_{state.index}"
    arguments_text = (
        state.final_arguments or state.arguments or "".join(state.arguments_parts)
    )
    parse_input = parser or (lambda text: {"arguments": text} if text else {})
    input_payload = parse_input(arguments_text)

    return anthropic_models.ToolUseBlock(
        type="tool_use",
        id=str(tool_id),
        name=str(state.name or "tool"),
        input=input_payload,
    )


def emit_anthropic_tool_use_events(
    index: int,
    state: ToolCallState,
    *,
    parser: Callable[[str], dict[str, Any]] | None = None,
) -> list[anthropic_models.MessageStreamEvent]:
    """Build start/stop events for a tool-use block at the given index."""

    block = build_anthropic_tool_use_block(
        state,
        default_id=f"call_{state.index}",
        parser=parser,
    )
    return [
        anthropic_models.ContentBlockStartEvent(
            type="content_block_start", index=index, content_block=block
        ),
        anthropic_models.ContentBlockStopEvent(type="content_block_stop", index=index),
    ]


__all__ = [
    "ReasoningBuffer",
    "ReasoningPartState",
    "ToolCallState",
    "ToolCallTracker",
    "IndexedToolCallTracker",
    "ObfuscationTokenFactory",
    "build_anthropic_tool_use_block",
    "emit_anthropic_tool_use_events",
]
