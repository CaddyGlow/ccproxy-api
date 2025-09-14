from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
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
      - Parses <thinking signature="…">…</thinking> into Anthropic `ThinkingBlock`
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

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        """Convert request using typed models - delegate to ResponsesRequest to Anthropic adapter."""
        from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
            OpenAIResponsesRequestToAnthropicMessagesAdapter,
        )

        # Use the dedicated adapter for the transformation
        adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()
        return await adapter.adapt_request(request)

    async def adapt_response(
        self, response: BaseModel
    ) -> AnthropicMessageResponse:
        """Convert ResponseObject to AnthropicMessageResponse using typed models."""
        if not isinstance(response, openai_models.ResponseObject):
            raise ValueError(f"Expected ResponseObject, got {type(response)}")

        return await self._convert_response(response)

    async def _convert_response(
        self, response: openai_models.ResponseObject
    ) -> AnthropicMessageResponse:
        """Convert ResponseObject to AnthropicMessageResponse using typed models."""
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

        content_blocks: list[
            AnthropicTextBlock | AnthropicThinkingBlock | AnthropicToolUseBlock
        ] = []

        # Gather reasoning summaries (map to Anthropic ThinkingBlock)
        for item in response.output or []:
            # Handle both object attributes and dictionary keys
            item_type = None
            if hasattr(item, "type"):
                item_type = getattr(item, "type", None)
            elif isinstance(item, dict):
                item_type = item.get("type")

            if item_type == "reasoning":
                summary_parts: list[Any] = []
                if hasattr(item, "summary"):
                    summary_parts = getattr(item, "summary", []) or []
                elif isinstance(item, dict):
                    summary_parts = item.get("summary", []) or []

                texts: list[str] = []
                for p in summary_parts:
                    p_type = None
                    if hasattr(p, "type"):
                        p_type = getattr(p, "type", None)
                    elif isinstance(p, dict):
                        p_type = p.get("type")

                    if p_type == "summary_text":
                        text = None
                        if hasattr(p, "text"):
                            text = p.text
                        elif isinstance(p, dict):
                            text = p.get("text")
                        if text and isinstance(text, str):
                            texts.append(text)

                if texts:
                    content_blocks.append(
                        AnthropicThinkingBlock(
                            type="thinking",
                            thinking=" ".join(texts),
                            signature="",
                        )
                    )

        # Take first message output and process content
        for item in response.output or []:
            # Handle both object attributes and dictionary keys
            item_type = None
            if hasattr(item, "type"):
                item_type = getattr(item, "type", None)
            elif isinstance(item, dict):
                item_type = item.get("type")

            if item_type == "message":
                content_list: list[Any] = []
                if hasattr(item, "content"):
                    content_list = getattr(item, "content", []) or []
                elif isinstance(item, dict):
                    content_list = item.get("content", []) or []
                for part in content_list:
                    # Handle typed OutputTextContent objects
                    if (
                        hasattr(part, "type")
                        and getattr(part, "type", None) == "output_text"
                    ):
                        text = getattr(part, "text", "") or ""
                        # Extract thinking blocks using regex
                        last_idx = 0
                        for m in THINKING_PATTERN.finditer(text):
                            # Add text before thinking
                            if m.start() > last_idx:
                                prefix = text[last_idx : m.start()]
                                if prefix.strip():
                                    content_blocks.append(
                                        AnthropicTextBlock(type="text", text=prefix)
                                    )
                            # Add thinking block
                            signature = m.group(1) or ""
                            thinking_text = m.group(2) or ""
                            content_blocks.append(
                                AnthropicThinkingBlock(
                                    type="thinking",
                                    thinking=thinking_text,
                                    signature=signature,
                                )
                            )
                            last_idx = m.end()
                        # Add remainder after last match
                        tail = text[last_idx:]
                        if tail.strip():
                            content_blocks.append(
                                AnthropicTextBlock(type="text", text=tail)
                            )
                    # Handle dict-based content (tool_use falls into this category)
                    elif isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "output_text":
                            text = part.get("text", "") or ""
                            # Extract thinking blocks using regex (same logic as above)
                            last_idx = 0
                            for m in THINKING_PATTERN.finditer(text):
                                # Add text before thinking
                                if m.start() > last_idx:
                                    prefix = text[last_idx : m.start()]
                                    if prefix.strip():
                                        content_blocks.append(
                                            AnthropicTextBlock(type="text", text=prefix)
                                        )
                                # Add thinking block
                                signature = m.group(1) or ""
                                thinking_text = m.group(2) or ""
                                content_blocks.append(
                                    AnthropicThinkingBlock(
                                        type="thinking",
                                        thinking=thinking_text,
                                        signature=signature,
                                    )
                                )
                                last_idx = m.end()
                            # Add remainder after last match
                            tail = text[last_idx:]
                            if tail.strip():
                                content_blocks.append(
                                    AnthropicTextBlock(type="text", text=tail)
                                )
                        elif part_type == "tool_use":
                            # Handle dict-based tool_use
                            tool_id = part.get("id", "tool_1") or "tool_1"
                            name = part.get("name", "function") or "function"
                            input_obj = (
                                part.get("arguments", part.get("input", {})) or {}
                            )
                            content_blocks.append(
                                AnthropicToolUseBlock(
                                    type="tool_use",
                                    id=tool_id,
                                    name=name,
                                    input=input_obj,
                                )
                            )
                    # Handle typed tool_use objects (fallback, unlikely to be used)
                    elif (
                        hasattr(part, "type")
                        and getattr(part, "type", None) == "tool_use"
                    ):
                        # Best-effort mapping if present in content
                        tool_id = getattr(part, "id", "tool_1") or "tool_1"
                        name = getattr(part, "name", "function") or "function"
                        input_obj = (
                            getattr(part, "arguments", getattr(part, "input", {})) or {}
                        )
                        content_blocks.append(
                            AnthropicToolUseBlock(
                                type="tool_use", id=tool_id, name=name, input=input_obj
                            )
                        )
                break

        # Create usage object
        anthropic_usage = AnthropicUsage(
            input_tokens=response.usage.input_tokens or 0 if response.usage else 0,
            output_tokens=response.usage.output_tokens or 0 if response.usage else 0,
        )

        return AnthropicMessageResponse(
            id=response.id or "msg_1",
            type="message",
            role="assistant",
            model=response.model or "",
            content=content_blocks,  # type: ignore[arg-type]
            stop_reason="end_turn",
            stop_sequence=None,
            usage=anthropic_usage,
        )

    def adapt_stream(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        """Convert OpenAI Response stream to Anthropic MessageStreamEvent stream."""
        return self._convert_stream(stream)

    async def _convert_stream(
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

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert OpenAI error to Anthropic error format using typed models."""
        from ccproxy.llms.anthropic.models import (
            APIError,
            ErrorType,
            InvalidRequestError,
            RateLimitError,
        )
        from ccproxy.llms.anthropic.models import (
            ErrorResponse as AnthropicErrorResponse,
        )
        from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

        # Error type mapping from OpenAI to Anthropic
        error_type_mapping = {
            "invalid_request_error": "invalid_request_error",
            "authentication_error": "invalid_request_error",
            "permission_error": "invalid_request_error",
            "not_found_error": "invalid_request_error",
            "rate_limit_error": "rate_limit_error",
            "internal_server_error": "api_error",
            "overloaded_error": "api_error",
        }

        # Handle OpenAI ErrorResponse format
        if isinstance(error, OpenAIErrorResponse):
            openai_error = error.error
            error_message = openai_error.message
            openai_error_type = openai_error.type or "api_error"

            # Map to Anthropic error type
            anthropic_error_type = error_type_mapping.get(
                openai_error_type, "api_error"
            )

            # Create appropriate Anthropic error model
            anthropic_error: ErrorType
            if anthropic_error_type == "invalid_request_error":
                anthropic_error = InvalidRequestError(message=error_message)
            elif anthropic_error_type == "rate_limit_error":
                anthropic_error = RateLimitError(message=error_message)
            else:
                anthropic_error = APIError(message=error_message)

            return AnthropicErrorResponse(error=anthropic_error)

        # Handle generic BaseModel errors or malformed errors
        if hasattr(error, "error") and hasattr(error.error, "message"):
            # Try to extract message from nested error structure
            error_message = error.error.message
            fallback_error: ErrorType = APIError(message=error_message)
            return AnthropicErrorResponse(error=fallback_error)

        # Fallback for unknown error formats
        error_message = "Unknown error occurred"
        if hasattr(error, "message"):
            error_message = error.message
        elif hasattr(error, "model_dump"):
            # Try to extract any available message from model dump
            error_dict = error.model_dump()
            error_message = str(error_dict.get("message", error_dict))

        generic_error: ErrorType = APIError(message=error_message)
        return AnthropicErrorResponse(error=generic_error)
