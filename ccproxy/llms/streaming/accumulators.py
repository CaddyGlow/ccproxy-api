"""Stream accumulators for different LLM streaming formats.

These accumulators process streaming response chunks and rebuild complete response objects
with all elements like content blocks, tool calls, thinking/reasoning, etc.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from pydantic import TypeAdapter, ValidationError

from ccproxy.llms.models import openai as openai_models


logger = structlog.get_logger(__name__)


_RESPONSES_STREAM_EVENT_ADAPTER = TypeAdapter(openai_models.AnyStreamEvent)
_RESPONSE_OBJECT_ADAPTER = TypeAdapter(openai_models.ResponseObject)


class StreamAccumulator:
    """Base class for accumulating streaming response chunks."""

    def __init__(self) -> None:
        self.tools: dict[str, dict[str, Any]] = {}
        self.content_blocks: list[dict[str, Any]] = []
        self.current_content_block: str | None = None
        self.text_content: str = ""

    def accumulate(self, event_name: str, event_data: dict[str, Any]) -> None:
        """Accumulate streaming events.

        Args:
            event_name: Name of the event (e.g., 'content_block_start')
            event_data: Data associated with the event
        """
        raise NotImplementedError

    def get_complete_tool_calls(self) -> list[dict[str, Any]]:
        """Get complete tool calls accumulated so far.

        Returns:
            List of complete tool calls
        """
        raise NotImplementedError

    def rebuild_response_object(self, response: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the complete response object with accumulated content.

        This method takes a response object and rebuilds it to include all accumulated
        content like tool calls, content blocks, thinking/reasoning, etc.

        Args:
            response: The original response object

        Returns:
            The updated response with all accumulated content
        """
        raise NotImplementedError


class ClaudeAccumulator(StreamAccumulator):
    """Accumulate Anthropic/Claude streaming events."""

    def __init__(self) -> None:
        super().__init__()
        self._index_to_key: dict[str, str] = {}
        self.content_blocks: list[dict[str, Any]] = []
        self.content_block_map: dict[str, dict[str, Any]] = {}  # Maps block_id to block
        self.message_metadata: dict[str, Any] = {
            "id": None,
            "type": "message",
            "role": "assistant",
            "model": None,
        }
        self._usage: dict[str, int] = {}
        self.stop_reason: str | None = None

    def accumulate(self, event_name: str, event_data: dict[str, Any]) -> None:
        """Accumulate Claude streaming events.

        Processes Claude-specific event types like:
        - content_block_start
        - content_block_delta
        - content_block_stop

        Args:
            event_name: Name of the event
            event_data: Data associated with the event
        """
        if event_name == "message_start":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "message_start"
            ):
                message = event_data.get("message", {})
                if isinstance(message, dict):
                    self.message_metadata["id"] = (
                        message.get("id") or self.message_metadata["id"]
                    )
                    self.message_metadata["type"] = message.get("type", "message")
                    self.message_metadata["role"] = message.get("role", "assistant")
                    self.message_metadata["model"] = (
                        message.get("model") or self.message_metadata["model"]
                    )

                    usage = message.get("usage")
                    if isinstance(usage, dict):
                        self._merge_usage(usage)

        elif event_name == "message_delta":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "message_delta"
            ):
                delta = event_data.get("delta")
                if isinstance(delta, dict):
                    stop_reason = delta.get("stop_reason")
                    if isinstance(stop_reason, str):
                        self.stop_reason = stop_reason

                usage = event_data.get("usage")
                if isinstance(usage, dict):
                    self._merge_usage(usage)

        elif event_name == "message_stop":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "message_stop"
            ):
                # No additional fields required, but keep hook for completeness.
                pass

        if event_name == "content_block_start":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "content_block_start"
            ):
                block = event_data.get("content_block", {})
                if not isinstance(block, dict):
                    return

                index_value = str(event_data.get("index", 0))
                block_id = block.get("id") or f"block_{index_value}_{len(self.tools)}"
                self._index_to_key[index_value] = block_id

                # Store block based on its type
                block_type = block.get("type", "")

                if block_type == "tool_use":
                    input_payload = block.get("input")
                    order = len(self.tools)
                    self.tools[block_id] = {
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": input_payload
                        if isinstance(input_payload, dict)
                        else {},
                        "partial_json": "",
                        "index": order,
                        "order": order,
                        "type": "tool_use",
                    }

                # Save all content blocks for rebuilding the full response
                self.content_block_map[block_id] = {
                    "id": block.get("id", block_id),
                    "type": block_type,
                    "index": int(index_value),
                }

                # Add type-specific fields
                if block_type == "text":
                    self.content_block_map[block_id]["text"] = ""
                elif block_type == "tool_use":
                    self.content_block_map[block_id]["name"] = block.get("name")
                    self.content_block_map[block_id]["input"] = block.get("input", {})
                elif block_type == "thinking":
                    self.content_block_map[block_id]["thinking"] = ""
                    signature = block.get("signature")
                    if isinstance(signature, str) and signature:
                        self.content_block_map[block_id]["signature"] = signature

                # Set current content block for delta updates
                self.current_content_block = (
                    str(block_id) if block_id is not None else None
                )

        elif event_name == "content_block_delta":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "content_block_delta"
            ):
                index_value = str(event_data.get("index", 0))
                block_id = self._index_to_key.get(index_value)
                delta = event_data.get("delta", {})

                if block_id and isinstance(delta, dict):
                    # For tool use blocks
                    if (
                        delta.get("type") == "input_json_delta"
                        and block_id in self.tools
                    ):
                        self.tools[block_id]["partial_json"] += delta.get(
                            "partial_json", ""
                        )

                    # For text blocks
                    elif (
                        delta.get("type") in {"text_delta", "text"}
                        and block_id in self.content_block_map
                    ):
                        block = self.content_block_map[block_id]
                        if block.get("type") == "text":
                            block["text"] = block.get("text", "") + delta.get(
                                "text", ""
                            )
                            self.text_content += delta.get("text", "")

                    # For thinking blocks
                    elif (
                        delta.get("type") in {"thinking_delta", "thinking"}
                        and block_id in self.content_block_map
                    ):
                        block = self.content_block_map[block_id]
                        if block.get("type") == "thinking":
                            block["thinking"] = block.get("thinking", "") + delta.get(
                                "thinking", ""
                            )

        elif event_name == "content_block_stop":
            if (
                isinstance(event_data, dict)
                and event_data.get("type") == "content_block_stop"
            ):
                index_value = str(event_data.get("index", 0))
                block_id = self._index_to_key.get(index_value)

                # Finalize tool use blocks by parsing JSON
                if block_id in self.tools and self.tools[block_id]["partial_json"]:
                    try:
                        payload = self.tools[block_id]["partial_json"]
                        self.tools[block_id]["input"] = json.loads(payload)

                        # Also update in content block map
                        if block_id in self.content_block_map:
                            self.content_block_map[block_id]["input"] = json.loads(
                                payload
                            )
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "claude_tool_json_decode_failed",
                            error=str(exc),
                            raw=self.tools[block_id]["partial_json"],
                        )

                # Finalize the current content block and add to ordered list
                if block_id in self.content_block_map:
                    block = self.content_block_map[block_id]
                    if block not in self.content_blocks:
                        self.content_blocks.append(block)

    def get_complete_tool_calls(self) -> list[dict[str, Any]]:
        """Get complete tool calls accumulated so far.

        Returns:
            List of complete tool calls
        """
        complete: list[dict[str, Any]] = []

        for tool_data in self.tools.values():
            if tool_data.get("input") is None:
                continue

            complete.append(
                {
                    "id": tool_data.get("id"),
                    "type": "function",
                    "name": tool_data.get("name"),
                    "input": tool_data.get("input"),
                    "function": {
                        "name": tool_data.get("name"),
                        "arguments": json.dumps(
                            tool_data.get("input", {}), ensure_ascii=False
                        ),
                    },
                    "index": tool_data.get("index"),
                    "order": tool_data.get("order"),
                }
            )

        return complete

    def rebuild_response_object(self, response: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the complete Claude response with all accumulated content.

        Args:
            response: Original Claude response

        Returns:
            Rebuilt response with complete content
        """
        content_blocks: list[dict[str, Any]] = []
        if self.content_blocks:
            sorted_blocks = sorted(self.content_blocks, key=lambda x: x.get("index", 0))
            for block in sorted_blocks:
                block_type = block.get("type")
                if block_type == "text":
                    content_blocks.append(
                        {
                            "type": "text",
                            "text": block.get("text", ""),
                        }
                    )
                elif block_type == "tool_use":
                    entry = {
                        "type": "tool_use",
                        "id": block.get("id"),
                        "name": block.get("name"),
                        "input": block.get("input", {}),
                    }
                    content_blocks.append(
                        {k: v for k, v in entry.items() if v not in (None, "")}
                    )
                elif block_type == "thinking":
                    content_blocks.append(
                        {
                            "type": "thinking",
                            "thinking": block.get("thinking", ""),
                            "signature": block.get("signature", ""),
                        }
                    )

        usage_payload = {
            "input_tokens": int(self._usage.get("input_tokens", 0)),
            "output_tokens": int(self._usage.get("output_tokens", 0)),
        }
        if "cache_read_input_tokens" in self._usage:
            usage_payload["cache_read_input_tokens"] = int(
                self._usage.get("cache_read_input_tokens", 0)
            )
        else:
            usage_payload["cache_read_input_tokens"] = 0

        rebuilt: dict[str, Any] = {
            "id": self.message_metadata.get("id") or response.get("id"),
            "type": self.message_metadata.get("type", "message"),
            "role": self.message_metadata.get("role", "assistant"),
            "content": content_blocks,
            "model": self.message_metadata.get("model") or response.get("model"),
            "stop_reason": self.stop_reason or response.get("stop_reason"),
            "usage": usage_payload,
        }

        if self.text_content:
            rebuilt["text"] = self.text_content

        return rebuilt

    def get_block_info(self, index: int) -> tuple[str, dict[str, Any]] | None:
        """Return (block_id, block_data) for a content block index."""

        if index < 0:
            return None

        block_id = self._index_to_key.get(str(index))
        if not block_id:
            return None

        block = self.content_block_map.get(block_id)
        if block is None:
            return None

        return block_id, block

    def get_tool_entry(
        self,
        identifier: int | str,
    ) -> dict[str, Any] | None:
        """Fetch the tool metadata tracked by the accumulator.

        Args:
            identifier: Either the integer index from the stream event or the
                underlying block identifier tracked by the accumulator.

        Returns:
            The tracked tool entry if present.
        """

        block_id: str | None
        if isinstance(identifier, int):
            info = self.get_block_info(identifier)
            block_id = info[0] if info else None
        else:
            block_id = identifier

        if not block_id:
            return None

        return self.tools.get(block_id)

    def _merge_usage(self, usage: dict[str, Any]) -> None:
        for key, value in usage.items():
            if isinstance(value, int | float):
                self._usage[key] = int(value)


class OpenAIAccumulator(StreamAccumulator):
    """Accumulate tool calls emitted via OpenAI chat/completion deltas."""

    def __init__(self) -> None:
        super().__init__()
        # Track the most recent entry key per choice index so anonymous deltas
        # append to the correct in-flight tool call instead of creating a new slot.
        self._index_to_key: dict[str, str] = {}
        self.choices: dict[int, dict[str, Any]] = {}
        self.message_content: dict[int, str] = {}

    def accumulate(self, event_name: str, event_data: dict[str, Any]) -> None:
        """Accumulate OpenAI streaming events.

        Args:
            event_name: Name of the event
            event_data: Data associated with the event
        """
        if not isinstance(event_data, dict) or "choices" not in event_data:
            return

        for choice in event_data.get("choices", []):
            if not isinstance(choice, dict):
                continue

            # Track choice index
            choice_index = choice.get("index", 0)

            # Initialize choice if not already tracked
            if choice_index not in self.choices:
                self.choices[choice_index] = {
                    "index": choice_index,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": None,
                }
                self.message_content[choice_index] = ""

            # Update finish reason if provided
            if "finish_reason" in choice:
                self.choices[choice_index]["finish_reason"] = choice["finish_reason"]

            # Update message content if provided
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue

            # Update message role if provided
            if "role" in delta:
                self.choices[choice_index]["message"]["role"] = delta["role"]

            # Update message content if provided
            if "content" in delta and delta["content"] is not None:
                content = delta["content"]
                self.message_content[choice_index] += content
                self.choices[choice_index]["message"]["content"] += content
                self.text_content += content

            # Process tool calls
            if "tool_calls" not in delta:
                continue

            for tool_call in delta.get("tool_calls", []) or []:
                if not isinstance(tool_call, dict):
                    continue

                index = int(tool_call.get("index", 0))
                index_key = str(index)

                previous_key = self._index_to_key.get(index_key)
                tool_id = tool_call.get("id")
                if isinstance(tool_id, str) and tool_id:
                    key = tool_id
                else:
                    key = previous_key or f"call_{index}"

                self._index_to_key[index_key] = key

                migrated_entry = None
                if previous_key and previous_key != key:
                    migrated_entry = self.tools.pop(previous_key, None)

                entry = self.tools.get(key)
                if entry is None:
                    if migrated_entry is not None:
                        entry = migrated_entry
                    else:
                        entry = {
                            "id": None,
                            "type": None,
                            "function": {"name": None, "arguments": ""},
                            "index": index,
                            "order": len(self.tools),
                        }
                    self.tools[key] = entry

                entry.setdefault("function", {"name": None, "arguments": ""})
                entry.setdefault("order", len(self.tools))
                entry["index"] = index

                if isinstance(tool_id, str) and tool_id:
                    entry["id"] = tool_id
                elif not entry.get("id"):
                    entry["id"] = key

                if "type" in tool_call:
                    entry["type"] = tool_call["type"]

                function = tool_call.get("function", {})
                if isinstance(function, dict):
                    if "name" in function:
                        name_value = function["name"]
                        if name_value:
                            entry["function"]["name"] = name_value
                    if "arguments" in function:
                        entry["function"]["arguments"] += function["arguments"]

    def get_complete_tool_calls(self) -> list[dict[str, Any]]:
        """Get complete tool calls accumulated so far.

        Returns:
            List of complete tool calls
        """
        complete: list[dict[str, Any]] = []

        for call_data in self.tools.values():
            arguments = call_data["function"].get("arguments")
            if not arguments:
                continue

            complete.append(
                {
                    "id": call_data.get("id"),
                    "type": call_data.get("type"),
                    "index": call_data.get("index"),
                    "order": call_data.get("order"),
                    "function": {
                        "name": call_data["function"].get("name"),
                        "arguments": arguments,
                    },
                }
            )

        return complete

    def rebuild_response_object(self, response: dict[str, Any]) -> dict[str, Any]:
        """Rebuild the complete OpenAI response with all accumulated content.

        Args:
            response: Original OpenAI response

        Returns:
            Rebuilt response with complete content
        """
        # Create a copy of the original response
        rebuilt = dict(response)

        # Rebuild choices with accumulated data
        if self.choices:
            # Convert choices dict to list and sort by index
            choice_list = list(self.choices.values())
            choice_list.sort(key=lambda x: x.get("index", 0))

            # Update choices in the response
            rebuilt["choices"] = choice_list

            # Update messages with tool calls
            tool_calls = self.get_complete_tool_calls()
            if tool_calls:
                # Add tool calls to each choice's message
                for choice in rebuilt["choices"]:
                    if "message" in choice:
                        choice["message"]["tool_calls"] = tool_calls

        return rebuilt


class ResponsesAccumulator(StreamAccumulator):
    """Accumulate events emitted by the OpenAI Responses API using typed models."""

    def __init__(self) -> None:
        super().__init__()
        self._items: dict[str, openai_models.OutputItem] = {}
        self._items_by_index: dict[int, str] = {}
        self._text_fragments: dict[tuple[str, int], list[str]] = {}
        self._reasoning_summary: dict[
            str, dict[int, openai_models.ReasoningSummaryPart]
        ] = {}
        self._reasoning_text: dict[tuple[str, int], list[str]] = {}
        self._function_arguments: dict[str, list[str]] = {}
        self._latest_response: openai_models.ResponseObject | None = None
        self.completed_response: openai_models.ResponseObject | None = None
        self._sequence_counter = 0

    def accumulate(
        self,
        event_name: str,
        event_data: dict[str, Any] | openai_models.BaseStreamEvent,
    ) -> None:
        """Accumulate Responses API streaming events."""

        event = self._coerce_stream_event(event_name, event_data)
        if event is None:
            return

        if isinstance(event, openai_models.ResponseCreatedEvent):
            self._latest_response = event.response
            return

        if isinstance(event, openai_models.ResponseInProgressEvent):
            self._latest_response = event.response
            return

        if isinstance(event, openai_models.ResponseCompletedEvent):
            self.completed_response = event.response
            return

        if isinstance(event, openai_models.ResponseOutputItemAddedEvent):
            self._record_output_item(event.output_index, event.item)
            return

        if isinstance(event, openai_models.ResponseOutputItemDoneEvent):
            self._merge_output_item(event.output_index, event.item)
            return

        if isinstance(event, openai_models.ResponseOutputTextDeltaEvent):
            self._accumulate_text_delta(
                item_id=event.item_id,
                content_index=event.content_index,
                delta=event.delta,
            )
            return

        if isinstance(event, openai_models.ResponseOutputTextDoneEvent):
            self._finalize_text(
                item_id=event.item_id,
                content_index=event.content_index,
                text=event.text,
            )
            return

        if isinstance(event, openai_models.ResponseFunctionCallArgumentsDeltaEvent):
            self._accumulate_function_arguments(event.item_id, event.delta)
            return

        if isinstance(event, openai_models.ResponseFunctionCallArgumentsDoneEvent):
            self._finalize_function_arguments(event.item_id, event.arguments)
            return

        if isinstance(event, openai_models.ReasoningSummaryPartAddedEvent):
            self._store_reasoning_summary_part(
                item_id=event.item_id,
                summary_index=event.summary_index,
                part=event.part,
            )
            return

        if isinstance(event, openai_models.ReasoningSummaryPartDoneEvent):
            self._store_reasoning_summary_part(
                item_id=event.item_id,
                summary_index=event.summary_index,
                part=event.part,
            )
            return

        if isinstance(event, openai_models.ReasoningSummaryTextDeltaEvent):
            self._accumulate_reasoning_text(
                item_id=event.item_id,
                summary_index=event.summary_index,
                delta=event.delta,
            )
            return

        if isinstance(event, openai_models.ReasoningSummaryTextDoneEvent):
            self._finalize_reasoning_text(
                item_id=event.item_id,
                summary_index=event.summary_index,
                text=event.text,
            )
            return

    def get_complete_tool_calls(self) -> list[dict[str, Any]]:
        """Get complete tool calls accumulated so far."""

        complete: list[dict[str, Any]] = []
        for item in self._items.values():
            if item.type != "function_call":
                continue
            arguments = self._get_function_arguments(item.id)
            if not (item.name and arguments):
                continue
            if item.status and item.status != "completed":
                continue

            complete.append(
                {
                    "id": item.id,
                    "type": "function_call",
                    "call_id": item.call_id,
                    "function": {
                        "name": item.name,
                        "arguments": arguments,
                    },
                }
            )

        return complete

    def rebuild_response_object(self, response: dict[str, Any]) -> dict[str, Any]:
        """Rebuild a complete Responses API payload with accumulated data."""

        base_response = self.completed_response or self._latest_response
        response_model = self._coerce_response_object(base_response or response)
        if response_model is None:
            response_model = openai_models.ResponseObject(
                id=str(response.get("id", "response")),
                created_at=int(response.get("created_at", 0)),
                status=str(response.get("status", "completed")),
                model=str(response.get("model", "")),
                output=[],
                parallel_tool_calls=bool(response.get("parallel_tool_calls", False)),
            )

        outputs = self._build_outputs()
        if outputs:
            response_model = response_model.model_copy(update={"output": outputs})

        function_calls = self.get_complete_tool_calls()
        reasoning_summary = self._build_reasoning_summary()

        payload = response_model.model_dump()

        if function_calls:
            payload["tool_calls"] = function_calls

        if not reasoning_summary:
            fallback_summary: list[dict[str, Any]] = []
            for output_entry in payload.get("output", []):
                if not isinstance(output_entry, dict):
                    continue
                if output_entry.get("type") != "reasoning":
                    continue
                summary_list = output_entry.get("summary")
                if isinstance(summary_list, list):
                    for part in summary_list:
                        if isinstance(part, dict):
                            fallback_summary.append(part)
            if fallback_summary:
                reasoning_summary = fallback_summary

        if reasoning_summary:
            reasoning_obj = payload.get("reasoning") or {}
            reasoning_obj["summary"] = reasoning_summary
            payload["reasoning"] = reasoning_obj

        if self.text_content:
            payload["text"] = self.text_content

        return payload

    def get_completed_response(self) -> dict[str, Any] | None:
        """Return the final response payload captured from the stream, if any."""

        if isinstance(self.completed_response, openai_models.ResponseObject):
            return self.completed_response.model_dump()
        return None

    def _coerce_stream_event(
        self,
        event_name: str,
        event_data: dict[str, Any] | openai_models.BaseStreamEvent,
    ) -> openai_models.BaseStreamEvent | openai_models.ErrorEvent | None:
        if isinstance(event_data, openai_models.BaseStreamEvent):
            # Update sequence counter for events that have sequence_number
            self._sequence_counter = max(
                self._sequence_counter, event_data.sequence_number
            )
            return event_data
        # Special handling for ErrorEvent which doesn't inherit from BaseStreamEvent
        elif isinstance(event_data, openai_models.ErrorEvent):
            return event_data

        if not isinstance(event_data, dict):
            return None

        payload = dict(event_data)
        payload.setdefault("type", event_name)
        if "sequence_number" not in payload:
            self._sequence_counter += 1
            payload["sequence_number"] = self._sequence_counter

        try:
            wrapper = _RESPONSES_STREAM_EVENT_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            logger.debug(
                "responses_accumulator_invalid_event",
                event_type=event_name,
                error=str(exc),
            )
            return None

        event = wrapper.root
        # Only update sequence counter if the event has sequence_number
        # ErrorEvent doesn't inherit from BaseStreamEvent and lacks this attribute
        if hasattr(event, "sequence_number"):
            self._sequence_counter = max(self._sequence_counter, event.sequence_number)
        return event

    def _record_output_item(
        self, output_index: int, item: openai_models.OutputItem
    ) -> None:
        self._items[item.id] = item
        self._items_by_index[output_index] = item.id
        if item.text:
            self.text_content = item.text

    def _merge_output_item(
        self, output_index: int, item: openai_models.OutputItem
    ) -> None:
        existing = self._items.get(item.id)
        if existing is not None:
            merged = existing.model_copy(update=item.model_dump(exclude_unset=True))
        else:
            merged = item
        self._items[item.id] = merged
        self._items_by_index[output_index] = item.id
        if merged.text:
            self.text_content = merged.text

    def _accumulate_text_delta(
        self, *, item_id: str, content_index: int, delta: str
    ) -> None:
        key = (item_id, content_index)
        fragments = self._text_fragments.setdefault(key, [])
        fragments.append(delta)
        combined = "".join(fragments)
        self._update_output_item_text(item_id, combined)

    def _finalize_text(self, *, item_id: str, content_index: int, text: str) -> None:
        key = (item_id, content_index)
        fragments = self._text_fragments.get(key, [])
        final_text = text or "".join(fragments)
        self._update_output_item_text(item_id, final_text)

    def _update_output_item_text(self, item_id: str, text: str) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        updated = item.model_copy(update={"text": text})
        self._items[item_id] = updated
        self.text_content = text

    def _accumulate_function_arguments(self, item_id: str, delta: str) -> None:
        args = self._function_arguments.setdefault(item_id, [])
        args.append(delta)
        combined = "".join(args)
        self._update_output_item_arguments(item_id, combined)

    def _finalize_function_arguments(self, item_id: str, arguments: str) -> None:
        if arguments:
            self._function_arguments[item_id] = [arguments]
            self._update_output_item_arguments(item_id, arguments)

    def _update_output_item_arguments(self, item_id: str, arguments: str) -> None:
        item = self._items.get(item_id)
        if item is None:
            return
        updated = item.model_copy(
            update={"arguments": arguments, "status": item.status or "completed"}
        )
        self._items[item_id] = updated

    def _store_reasoning_summary_part(
        self,
        *,
        item_id: str,
        summary_index: int,
        part: openai_models.ReasoningSummaryPart,
    ) -> None:
        entry = self._reasoning_summary.setdefault(item_id, {})
        entry[summary_index] = part

    def _accumulate_reasoning_text(
        self, *, item_id: str, summary_index: int, delta: str
    ) -> None:
        key = (item_id, summary_index)
        fragments = self._reasoning_text.setdefault(key, [])
        fragments.append(delta)
        text_value = "".join(fragments)
        part = self._reasoning_summary.setdefault(item_id, {}).get(summary_index)
        if part is not None:
            self._reasoning_summary[item_id][summary_index] = part.model_copy(
                update={"text": text_value}
            )
        else:
            self._reasoning_summary.setdefault(item_id, {})[summary_index] = (
                openai_models.ReasoningSummaryPart(type="summary_text", text=text_value)
            )

    def _finalize_reasoning_text(
        self, *, item_id: str, summary_index: int, text: str
    ) -> None:
        final_text = text or "".join(
            self._reasoning_text.get((item_id, summary_index), [])
        )
        part = self._reasoning_summary.setdefault(item_id, {}).get(summary_index)
        if part is not None:
            self._reasoning_summary[item_id][summary_index] = part.model_copy(
                update={"text": final_text}
            )
        else:
            self._reasoning_summary[item_id][summary_index] = (
                openai_models.ReasoningSummaryPart(type="summary_text", text=final_text)
            )

    def _get_function_arguments(self, item_id: str) -> str | None:
        explicit = self._items.get(item_id)
        if explicit and explicit.arguments:
            return explicit.arguments
        fragments = self._function_arguments.get(item_id)
        if not fragments:
            return None
        return "".join(fragments)

    def _coerce_response_object(
        self, response: dict[str, Any] | openai_models.ResponseObject | None
    ) -> openai_models.ResponseObject | None:
        if isinstance(response, openai_models.ResponseObject):
            return response
        if not isinstance(response, dict):
            return None

        payload = dict(response)
        payload.setdefault("object", "response")
        payload.setdefault("created_at", int(payload.get("created_at") or 0))
        payload.setdefault("status", payload.get("status") or "completed")
        payload.setdefault("model", payload.get("model") or "")
        if isinstance(payload.get("output"), dict):
            payload["output"] = [payload["output"]]
        payload.setdefault("output", payload.get("output") or [])
        payload.setdefault(
            "parallel_tool_calls", payload.get("parallel_tool_calls", False)
        )

        try:
            return _RESPONSE_OBJECT_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            logger.debug(
                "responses_accumulator_response_normalization_failed",
                error=str(exc),
            )
            return openai_models.ResponseObject(
                id=str(payload.get("id") or "response"),
                created_at=int(payload.get("created_at") or 0),
                status=str(payload.get("status") or "completed"),
                model=str(payload.get("model") or ""),
                output=[],
                parallel_tool_calls=bool(payload.get("parallel_tool_calls") or False),
            )

    def _build_outputs(
        self,
    ) -> list[
        openai_models.MessageOutput
        | openai_models.ReasoningOutput
        | openai_models.FunctionCallOutput
        | dict[str, Any]
    ]:
        outputs: list[
            openai_models.MessageOutput
            | openai_models.ReasoningOutput
            | openai_models.FunctionCallOutput
            | dict[str, Any]
        ] = []

        for index in sorted(self._items_by_index):
            item_id = self._items_by_index[index]
            item = self._items.get(item_id)
            if item is None:
                continue

            if item.type == "function_call":
                outputs.append(
                    openai_models.FunctionCallOutput(
                        type="function_call",
                        id=item.id,
                        status=item.status or "completed",
                        name=item.name,
                        call_id=item.call_id,
                        arguments=self._get_function_arguments(item.id),
                    )
                )
                continue

            if item.type == "reasoning":
                summary_map = self._reasoning_summary.get(item.id, {})
                summary_entries: list[dict[str, Any]] = []
                for key in sorted(summary_map):
                    summary_part = summary_map[key]
                    summary_entries.append(summary_part.model_dump())
                if not summary_entries and item.summary:
                    for part in item.summary:
                        if hasattr(part, "model_dump"):
                            summary_entries.append(part.model_dump())
                        else:
                            summary_entries.append(part)
                outputs.append(
                    openai_models.ReasoningOutput(
                        type="reasoning",
                        id=item.id,
                        status=item.status or "completed",
                        summary=summary_entries or item.summary,
                    )
                )
                continue

            text_value = item.text or self._combined_text(item.id)
            content_entries: list[Any] = []
            if text_value:
                content_entries.append(
                    openai_models.OutputTextContent(type="output_text", text=text_value)
                )
            elif item.content:
                content_entries.extend(item.content)

            outputs.append(
                openai_models.MessageOutput(
                    type="message",
                    id=item.id,
                    status=item.status or "completed",
                    role="assistant"
                    if item.role is None or item.role not in ("assistant", "user")
                    else ("assistant" if item.role != "user" else "user"),
                    content=[
                        part.model_dump()
                        if isinstance(part, openai_models.OutputTextContent)
                        else part
                        for part in content_entries
                    ],
                )
            )

        return outputs

    def _combined_text(self, item_id: str) -> str | None:
        values: list[str] = []
        for (candidate_id, _), fragments in self._text_fragments.items():
            if candidate_id == item_id:
                values.extend(fragments)
        if values:
            return "".join(values)
        return None

    def _build_reasoning_summary(self) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for item_id, parts in self._reasoning_summary.items():
            item = self._items.get(item_id)
            status = item.status if item else "completed"
            for key in sorted(parts):
                part = parts[key]
                entry = part.model_dump()
                entry.setdefault("status", status)
                summary.append(entry)
        return summary
