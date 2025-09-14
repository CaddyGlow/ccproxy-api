from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.anthropic.models import (
    ThinkingBlock as AnthropicThinkingBlock,
)
from ccproxy.llms.anthropic.models import (
    ToolUseBlock as AnthropicToolUseBlock,
)
from ccproxy.llms.anthropic.models import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai import models as openai_models


THINKING_PATTERN = re.compile(
    r'<thinking(?:\s+signature="([^"]*)")?>(.*?)</thinking>', re.DOTALL
)


class OpenAIResponsesToAnthropicAdapter(
    BaseAPIAdapter[
        BaseModel,
        BaseModel,
        openai_models.AnyStreamEvent,
    ]
):
    """OpenAI Responses → Anthropic Messages adapter (non-streaming + streaming subset).

    Implemented
    - Non-streaming:
      - Parses <thinking signature=\"…\">…</thinking> into Anthropic `ThinkingBlock`
      - Emits remaining text as `TextBlock`
      - Maps basic `tool_use` items in content to `ToolUseBlock`
      - Maps usage fields (input/output tokens)
    - Streaming (basic):
      - Emits `message_start` → text `content_block_*` deltas → `message_stop`
      - Removes thinking tags and content from deltas (not supported in Anthropic SSE)

    TODO
    - Expand streaming to support more event types (function calls, web/code/file search)
    - Capture refusal/status details and propagate as message metadata if needed
    - Map stop reasons more precisely based on Responses status
    """

    def __init__(self) -> None:
        super().__init__(name="openai_responses_to_anthropic")

    # Minimal implementations for abstract methods - delegate to dict-based logic
    def _dict_to_request_model(self, request: dict[str, Any]) -> BaseModel:
        return BaseModel(**request)  # Minimal implementation

    def _dict_to_response_model(self, response: dict[str, Any]) -> BaseModel:
        return BaseModel(**response)  # Minimal implementation

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        return BaseModel(**error)  # Minimal implementation

    def _dict_stream_to_typed_stream(
        self,
        stream: AsyncIterator[dict[str, Any]],
    ) -> AsyncIterator[openai_models.AnyStreamEvent]:
        event_adapter = TypeAdapter(openai_models.AnyStreamEvent)

        async def generator() -> AsyncIterator[openai_models.AnyStreamEvent]:
            async for item in stream:
                try:
                    yield event_adapter.validate_python(item)
                except Exception:
                    continue

        return generator()

    async def adapt_request_typed(self, request: BaseModel) -> BaseModel:
        request_dict = (
            request.model_dump() if hasattr(request, "model_dump") else dict(request)
        )
        result_dict = await self.adapt_request(request_dict)
        return BaseModel(**result_dict)

    async def adapt_response_typed(self, response: BaseModel) -> BaseModel:
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )
        result_dict = await self.adapt_response(response_dict)
        return BaseModel(**result_dict)

    def adapt_stream_typed(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        """Convert OpenAI Response stream to Anthropic MessageStreamEvent stream."""
        return self._convert_stream_typed(stream)

    async def _convert_stream_typed(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        message_started = False
        index = 0
        for_event_id = "msg_stream"
        func_args_buffer: dict[int, list[str]] = {}
        tool_info: dict[int, dict[str, str]] = {}
        reasoning_buffer: list[str] = []
        text_block_started = False

        async for evt_wrapper in stream:
            if not hasattr(evt_wrapper, "root"):
                continue
            evt = evt_wrapper.root
            if not hasattr(evt, "type"):
                continue

            if (
                evt.type in ("response.created", "response.in_progress")
                and hasattr(evt, "response")
                and not message_started
            ):
                yield anthropic_models.MessageStartEvent(
                    type="message_start",
                    message=anthropic_models.MessageResponse(
                        id=for_event_id,
                        type="message",
                        role="assistant",
                        content=[],
                        model=evt.response.model or "",
                        stop_reason=None,
                        stop_sequence=None,
                        usage=anthropic_models.Usage(input_tokens=0, output_tokens=0),
                    ),
                )
                message_started = True
            elif evt.type == "response.output_text.delta":
                text = evt.delta or ""
                if text:
                    if not text_block_started:
                        yield anthropic_models.ContentBlockStartEvent(
                            type="content_block_start",
                            index=index,
                            content_block=anthropic_models.TextBlock(
                                type="text", text=""
                            ),
                        )
                        text_block_started = True
                    yield anthropic_models.ContentBlockDeltaEvent(
                        type="content_block_delta",
                        index=index,
                        delta=anthropic_models.TextBlock(type="text", text=text),
                    )
            elif evt.type == "response.reasoning_summary_text.delta":
                delta = evt.delta or ""
                if isinstance(delta, str):
                    reasoning_buffer.append(delta)
            elif evt.type == "response.reasoning_summary_text.done":
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=index
                    )
                    text_block_started = False
                    index += 1
                summary = "".join(reasoning_buffer)
                if summary:
                    yield anthropic_models.ContentBlockStartEvent(
                        type="content_block_start",
                        index=index,
                        content_block=anthropic_models.ThinkingBlock(
                            type="thinking",
                            thinking=summary,
                            signature="",
                        ),
                    )
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=index
                    )
                    index += 1
                reasoning_buffer.clear()
            elif evt.type == "response.function_call_arguments.delta":
                output_index = evt.output_index
                if output_index not in func_args_buffer:
                    func_args_buffer[output_index] = []
                    tool_info[output_index] = {
                        "id": evt.item_id or f"call_{output_index}",
                        "name": evt.item_id or f"call_{output_index}",
                    }
                delta = evt.delta or ""
                if isinstance(delta, str):
                    func_args_buffer[output_index].append(delta)
            elif evt.type == "response.function_call_arguments.done":
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=index
                    )
                    text_block_started = False
                    index += 1
                output_index = evt.output_index
                try:
                    import json

                    args_str = evt.arguments or "".join(
                        func_args_buffer.get(output_index, [])
                    )
                    args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
                except Exception:
                    args_obj = {}

                info = tool_info.get(
                    output_index,
                    {"id": f"call_{output_index}", "name": f"call_{output_index}"},
                )

                yield anthropic_models.ContentBlockStartEvent(
                    type="content_block_start",
                    index=index,
                    content_block=anthropic_models.ToolUseBlock(
                        type="tool_use",
                        id=info["id"],
                        name=info["name"],
                        input=args_obj,
                    ),
                )
                yield anthropic_models.ContentBlockStopEvent(
                    type="content_block_stop", index=index
                )
                index += 1
                if output_index in func_args_buffer:
                    del func_args_buffer[output_index]
                if output_index in tool_info:
                    del tool_info[output_index]

            elif evt.type == "error":
                yield anthropic_models.ErrorEvent(
                    type="error",
                    error=anthropic_models.ErrorDetail(message=evt.error.message),
                )

            elif evt.type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                if text_block_started:
                    yield anthropic_models.ContentBlockStopEvent(
                        type="content_block_stop", index=index
                    )
                yield anthropic_models.MessageStopEvent(type="message_stop")
                break

    async def adapt_error_typed(self, error: BaseModel) -> BaseModel:
        error_dict = error.model_dump() if hasattr(error, "model_dump") else dict(error)
        result_dict = await self.adapt_error(error_dict)
        return BaseModel(**result_dict)

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        # Delegate ResponseRequest -> Anthropic Messages request
        from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
            OpenAIResponsesRequestToAnthropicMessagesAdapter,
        )

        return await OpenAIResponsesRequestToAnthropicMessagesAdapter().adapt_request(
            request
        )

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # response is expected to be an OpenAI ResponseObject dict
        output = response.get("output") or []
        content_blocks: list[dict[str, Any]] = []

        # Gather reasoning summaries (map to Anthropic ThinkingBlock)
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                summary_parts = item.get("summary") or []
                texts: list[str] = []
                for p in summary_parts:
                    if isinstance(p, dict) and p.get("type") == "summary_text":
                        t = p.get("text")
                        if isinstance(t, str):
                            texts.append(t)
                if texts:
                    content_blocks.append(
                        AnthropicThinkingBlock(
                            type="thinking",
                            thinking=" ".join(texts),
                            signature="",
                        ).model_dump()
                    )

        # Take first message output
        for item in output:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text") or ""
                    # Extract thinking blocks
                    last_idx = 0
                    for m in THINKING_PATTERN.finditer(text):
                        # text before thinking
                        if m.start() > last_idx:
                            prefix = text[last_idx : m.start()]
                            if prefix.strip():
                                content_blocks.append(
                                    AnthropicTextBlock(
                                        type="text", text=prefix
                                    ).model_dump()
                                )
                        signature = m.group(1) or ""
                        thinking_text = m.group(2) or ""
                        content_blocks.append(
                            AnthropicThinkingBlock(
                                type="thinking",
                                thinking=thinking_text,
                                signature=signature,
                            ).model_dump()
                        )
                        last_idx = m.end()
                    # Remainder after last match
                    tail = text[last_idx:]
                    if tail.strip():
                        content_blocks.append(
                            AnthropicTextBlock(type="text", text=tail).model_dump()
                        )
                elif isinstance(part, dict) and part.get("type") == "tool_use":
                    # Best-effort mapping if present in content
                    tool_id = part.get("id") or "tool_1"
                    name = part.get("name") or "function"
                    input_obj = part.get("arguments") or part.get("input") or {}
                    content_blocks.append(
                        AnthropicToolUseBlock(
                            type="tool_use", id=tool_id, name=name, input=input_obj
                        ).model_dump()
                    )
            break

        usage = response.get("usage") or {}
        anthropic_usage = AnthropicUsage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )

        payload = AnthropicMessageResponse(
            id=response.get("id") or "msg_1",
            type="message",
            role="assistant",
            model=response.get("model") or "",
            content=content_blocks,  # type: ignore[arg-type]
            stop_reason="end_turn",
            stop_sequence=None,
            usage=anthropic_usage,
        )
        return payload.model_dump()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
