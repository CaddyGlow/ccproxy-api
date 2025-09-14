from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
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


THINKING_PATTERN = re.compile(
    r'<thinking(?:\s+signature="([^"]*)")?>(.*?)</thinking>', re.DOTALL
)


class OpenAIResponsesToAnthropicAdapter(BaseAPIAdapter):
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
            role="assistant",
            model=response.get("model") or "",
            content=content_blocks,  # type: ignore[arg-type]
            stop_reason="end_turn",
            stop_sequence=None,
            usage=anthropic_usage,
        )
        return payload.model_dump()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def generator() -> AsyncGenerator[dict[str, Any], None]:
            # Basic text-only streaming mapping. Thinking content is skipped.
            message_started = False
            index = 0
            for_event_id = "msg_stream"
            func_args_buffer: list[str] = []
            tool_block_emitted = False

            async for evt in stream:
                etype = evt.get("type") if isinstance(evt, dict) else None
                if (
                    etype in ("response.created", "response.in_progress")
                    and not message_started
                ):
                    # Emit message_start with empty content and zero usage
                    yield {
                        "type": "message_start",
                        "message": {
                            "id": for_event_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": evt.get("response", {}).get("model", ""),
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        },
                    }
                    message_started = True
                elif etype == "response.output_text.delta":
                    raw = evt.get("delta") or ""
                    # Remove thinking tags and content
                    text_no_thinking = THINKING_PATTERN.sub("", raw)
                    if text_no_thinking:
                        if index == 0:
                            yield {
                                "type": "content_block_start",
                                "index": 0,
                                "content_block": {"type": "text", "text": ""},
                            }
                        yield {
                            "type": "content_block_delta",
                            "index": 0,
                            "delta": {"type": "text", "text": text_no_thinking},
                        }
                        index = 1
                elif etype == "response.function_call_arguments.delta":
                    delta = evt.get("delta") or ""
                    if isinstance(delta, str):
                        func_args_buffer.append(delta)
                elif etype == "response.function_call_arguments.done":
                    # Emit a tool_use content block with parsed arguments
                    try:
                        import json

                        args_str = evt.get("arguments") or "".join(func_args_buffer)
                        args_obj = json.loads(args_str) if isinstance(args_str, str) else {}
                    except Exception:
                        args_obj = {}
                    yield {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "function",
                            "input": args_obj,
                        },
                    }
                    yield {"type": "content_block_stop", "index": index}
                    tool_block_emitted = True
                    index = index + 1
                elif etype in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    if index > 0:
                        yield {"type": "content_block_stop", "index": 0}
                    # Emit message_stop at end
                    yield {"type": "message_stop"}
                    break

        return generator()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
