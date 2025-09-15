from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import (
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
    convert_anthropic_usage_to_openai_completion_usage,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.openai import models as openai_models


FinishReason = Literal["stop", "length", "tool_calls"]


class AnthropicMessagesToOpenAIChatAdapter(
    BaseAPIAdapter[
        anthropic_models.CreateMessageRequest,
        anthropic_models.MessageResponse,
        anthropic_models.MessageStreamEvent,
    ]
):
    """Anthropic Messages ↔ OpenAI Chat adapter (response and request subset).

    Implemented
    - adapt_response: Anthropic → OpenAI Chat response
      - Serializes `thinking` blocks to XML: <thinking signature="...">…</thinking>
      - Aggregates visible `text` into assistant message content
      - Maps stop_reason and usage to OpenAI fields
    - adapt_request: Anthropic → OpenAI Chat request (subset)
      - Maps `system` and `messages` (text + data‑URL images → `image_url` parts)
      - Maps custom tools → function tools
      - Maps tool_choice (auto/any/tool/none) and parallel control

    TODO
    - Rich content mapping: tool use/results in request content (rare for Chat)
    - Advanced message annotations, refusal reasons, and logprobs if needed
    - Streaming conversions are handled elsewhere; this class focuses on payloads
    """

    def __init__(self) -> None:
        super().__init__(name="anthropic_messages_to_openai_chat")

    # New strongly-typed methods
    async def adapt_request(self, request: BaseModel) -> BaseModel:
        """Convert Anthropic CreateMessageRequest to OpenAI ChatCompletionRequest."""
        if not isinstance(request, anthropic_models.CreateMessageRequest):
            raise ValueError(f"Expected CreateMessageRequest, got {type(request)}")

        return await self._convert_request(request)

    async def adapt_response(self, response: BaseModel) -> BaseModel:
        """Convert Anthropic MessageResponse to OpenAI ChatCompletionResponse."""
        if not isinstance(response, anthropic_models.MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return await self._convert_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        """Convert Anthropic MessageStreamEvent stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream(stream)

    async def _convert_stream(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        model_id = ""
        finish_reason: FinishReason = "stop"
        final_usage: BaseModel | None = None

        async for evt in stream:
            if not hasattr(evt, "type"):
                continue

            if evt.type == "message_start":
                model_id = evt.message.model or ""
            elif evt.type == "content_block_delta":
                text = evt.delta.text
                if text:
                    yield openai_models.ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=text
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
            elif evt.type == "message_delta":
                if evt.delta.stop_reason:
                    finish_reason = cast(
                        FinishReason,
                        ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
                            evt.delta.stop_reason, "stop"
                        ),
                    )
                final_usage = evt.usage
            elif evt.type == "message_stop":
                usage = None
                if final_usage is not None:
                    usage = cast(
                        openai_models.CompletionUsage,
                        convert_anthropic_usage_to_openai_completion_usage(final_usage),
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

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert Anthropic error to OpenAI error format."""
        from ccproxy.llms.adapters.mapping import convert_anthropic_error_to_openai

        return convert_anthropic_error_to_openai(error)

    # Implementation methods
    async def _convert_request(
        self, request: anthropic_models.CreateMessageRequest
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
                        if (
                            hasattr(block, "id")
                            and hasattr(block, "name")
                            and hasattr(block, "input")
                        ):
                            tool_calls.append(
                                {
                                    "id": block.id,
                                    "type": "function",
                                    "function": {
                                        "name": block.name,
                                        "arguments": str(block.input),
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
                    assistant_msg["content"] = (
                        " ".join(text_parts) if text_parts else None
                    )
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
                            if hasattr(block, "tool_use_id") and hasattr(
                                block, "content"
                            ):
                                openai_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": block.tool_use_id,
                                        "content": str(block.content),
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

    async def _convert_response(
        self, response: anthropic_models.MessageResponse
    ) -> openai_models.ChatCompletionResponse:
        """Convert Anthropic MessageResponse to OpenAI ChatCompletionResponse using typed models."""
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

        usage_model = convert_anthropic_usage_to_openai_completion_usage(response.usage)

        payload = {
            "id": response.id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": finish_reason,
                }
            ],
            "created": 0,
            "model": response.model,
            "object": "chat.completion",
            "usage": usage_model.model_dump(),
        }
        return openai_models.ChatCompletionResponse.model_validate(payload)


# Backward-compatible alias for tests
class AnthropicToOpenAIChatCompletionsAdapter(AnthropicMessagesToOpenAIChatAdapter):
    pass
