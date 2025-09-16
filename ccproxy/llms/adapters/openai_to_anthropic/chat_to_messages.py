from __future__ import annotations

import json
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shared import (
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
    DEFAULT_MAX_TOKENS,
    convert_openai_error_to_anthropic,
)
from ccproxy.llms.adapters.anthropic_to_openai.messages_to_chat import (
    convert_anthropic_message_to_chat_response,
)
from ccproxy.llms.anthropic.models import (
    CreateMessageRequest,
    MessageResponse,
    MessageStreamEvent,
)
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
)


FinishReason = Literal["stop", "length", "tool_calls"]


class OpenAIChatToAnthropicMessagesAdapter(
    BaseAPIAdapter[
        openai_models.ChatCompletionRequest,
        MessageResponse,
        MessageStreamEvent,
    ]
):
    """OpenAI Chat → Anthropic Messages request adapter.

    Implemented
    - model: passthrough
    - messages: extracts `system` and maps `user`/`assistant` to Anthropic `messages`
    - max_completion_tokens/max_tokens → `max_tokens` (with default fallback)
    - stream: passthrough
    - thinking: enables Anthropic `thinking` based on `reasoning_effort` (low/medium/high)
      and o1/o3 model families with budgets and safety (ensures `max_tokens > budget`),
      forces `temperature=1.0` when enabled
    - images: supports OpenAI `image_url` with data URL, mapped to Anthropic base64 `ImageBlock`
    - tools: maps OpenAI function tools → Anthropic custom tools
    - tool_choice: maps `none|auto|required|{type:function}` → Anthropic tool choice and
      sets `disable_parallel_tool_use` from `parallel_tool_calls`
    - response_format: injects system guidance for `json_object` and `json_schema`

    TODO
    - Full multimodal request support (audio/prompt images that aren't data URLs)
    - MCP servers, container fields, and advanced options not yet mapped
    - Stop sequences and more nuanced safety/service_tier mapping
    """

    def __init__(self) -> None:
        super().__init__(name="openai_chat_to_anthropic_messages")

    # New strongly-typed methods
    async def adapt_request(
        self, request: ChatCompletionRequest
    ) -> CreateMessageRequest:
        """Convert OpenAI ChatCompletionRequest to Anthropic CreateMessageRequest."""
        if not isinstance(request, ChatCompletionRequest):
            raise ValueError(f"Expected ChatCompletionRequest, got {type(request)}")

        return await self._convert_request(request)

    async def adapt_response(self, response: MessageResponse) -> BaseModel:
        """Convert Anthropic MessageResponse to OpenAI ChatCompletionResponse."""
        if not isinstance(response, MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return convert_anthropic_message_to_chat_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[MessageStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Convert Anthropic MessageStreamEvent stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream(stream)

    def _convert_stream(
        self, stream: AsyncIterator[MessageStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Convert Anthropic stream to OpenAI stream using typed models."""

        async def generator() -> AsyncGenerator[ChatCompletionChunk, None]:
            model_id = ""
            finish_reason: FinishReason = "stop"
            usage_prompt = 0
            usage_completion = 0

            async for evt in stream:
                if not hasattr(evt, "type"):
                    continue

                if evt.type == "message_start":
                    model_id = evt.message.model or ""
                elif evt.type == "content_block_delta":
                    text = evt.delta.text
                    if text:
                        yield ChatCompletionChunk(
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
                    usage_prompt = evt.usage.input_tokens
                    usage_completion = evt.usage.output_tokens
                elif evt.type == "message_stop":
                    usage = None
                    if usage_prompt or usage_completion:
                        usage = openai_models.CompletionUsage(
                            prompt_tokens=usage_prompt,
                            completion_tokens=usage_completion,
                            total_tokens=usage_prompt + usage_completion,
                        )
                    yield ChatCompletionChunk(
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

        return generator()

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert OpenAI error payloads to the Anthropic envelope."""
        return convert_openai_error_to_anthropic(error)

    async def _convert_request(
        self, request: ChatCompletionRequest
    ) -> CreateMessageRequest:
        """Convert OpenAI ChatCompletionRequest to Anthropic CreateMessageRequest using typed models."""
        model = request.model.strip() if request.model else ""

        # Determine max tokens
        max_tokens = request.max_completion_tokens
        if max_tokens is None:
            max_tokens = request.max_tokens
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        # Extract system message if present
        system_value: str | None = None
        out_messages: list[dict[str, Any]] = []

        for msg in request.messages or []:
            role = msg.role
            content = msg.content
            tool_calls = getattr(msg, "tool_calls", None)

            if role == "system":
                if isinstance(content, str):
                    system_value = content
                elif isinstance(content, list):
                    texts = [
                        part.text
                        for part in content
                        if hasattr(part, "type")
                        and part.type == "text"
                        and hasattr(part, "text")
                    ]
                    system_value = " ".join([t for t in texts if t]) or None
            elif role == "assistant":
                if tool_calls:
                    blocks = []
                    if content:  # Add text content if present
                        blocks.append({"type": "text", "text": str(content)})
                    for tc in tool_calls:
                        func_info = tc.function
                        tool_name = func_info.name if func_info else None
                        tool_args = func_info.arguments if func_info else "{}"
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": str(tool_name) if tool_name is not None else "",
                                "input": json.loads(str(tool_args)),
                            }
                        )
                    out_messages.append({"role": "assistant", "content": blocks})
                elif content is not None:
                    out_messages.append({"role": "assistant", "content": content})

            elif role == "tool":
                tool_call_id = getattr(msg, "tool_call_id", None)
                out_messages.append(
                    {
                        "role": "user",  # Anthropic uses 'user' role for tool results
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_call_id,
                                "content": str(content),
                            }
                        ],
                    }
                )
            elif role == "user":
                if content is None:
                    continue
                if isinstance(content, list):
                    user_blocks: list[dict[str, Any]] = []
                    text_accum: list[str] = []
                    for part in content:
                        # Handle both dict and Pydantic object inputs
                        if isinstance(part, dict):
                            ptype = part.get("type")
                            if ptype == "text":
                                t = part.get("text")
                                if isinstance(t, str):
                                    text_accum.append(t)
                            elif ptype == "image_url":
                                image_info = part.get("image_url")
                                if isinstance(image_info, dict):
                                    url = image_info.get("url")
                                    if isinstance(url, str) and url.startswith("data:"):
                                        try:
                                            header, b64data = url.split(",", 1)
                                            mediatype = header.split(";")[0].split(
                                                ":", 1
                                            )[1]
                                            user_blocks.append(
                                                {
                                                    "type": "image",
                                                    "source": {
                                                        "type": "base64",
                                                        "media_type": str(mediatype),
                                                        "data": str(b64data),
                                                    },
                                                }
                                            )
                                        except Exception:
                                            pass
                        elif hasattr(part, "type"):
                            # Pydantic object case
                            ptype = part.type
                            if ptype == "text" and hasattr(part, "text"):
                                t = part.text
                                if isinstance(t, str):
                                    text_accum.append(t)
                            elif ptype == "image_url" and hasattr(part, "image_url"):
                                url = part.image_url.url if part.image_url else None
                                if isinstance(url, str) and url.startswith("data:"):
                                    try:
                                        header, b64data = url.split(",", 1)
                                        mediatype = header.split(";")[0].split(":", 1)[
                                            1
                                        ]
                                        user_blocks.append(
                                            {
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": str(mediatype),
                                                    "data": str(b64data),
                                                },
                                            }
                                        )
                                    except Exception:
                                        pass
                    if user_blocks:
                        # If we have images, always use list format
                        if text_accum:
                            user_blocks.insert(
                                0, {"type": "text", "text": " ".join(text_accum)}
                            )
                        out_messages.append({"role": "user", "content": user_blocks})
                    elif len(text_accum) > 1:
                        # Multiple text parts - use list format
                        text_blocks = [{"type": "text", "text": " ".join(text_accum)}]
                        out_messages.append({"role": "user", "content": text_blocks})
                    elif len(text_accum) == 1:
                        # Single text part - use string format
                        out_messages.append({"role": "user", "content": text_accum[0]})
                    else:
                        # No content - use empty string
                        out_messages.append({"role": "user", "content": ""})
                else:
                    out_messages.append({"role": "user", "content": content})

        payload_data: dict[str, Any] = {
            "model": model,
            "messages": out_messages,
            "max_tokens": max_tokens,
        }

        # Inject system guidance for response_format JSON modes
        resp_fmt = request.response_format
        if resp_fmt is not None:
            inject: str | None = None
            if resp_fmt.type == "json_object":
                inject = (
                    "Respond ONLY with a valid JSON object. "
                    "Do not include any additional text, markdown, or explanation."
                )
            elif resp_fmt.type == "json_schema" and hasattr(resp_fmt, "json_schema"):
                schema = resp_fmt.json_schema
                try:
                    if schema is not None:
                        schema_str = json.dumps(
                            schema.model_dump()
                            if hasattr(schema, "model_dump")
                            else schema,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    else:
                        schema_str = "{}"
                except Exception:
                    schema_str = str(schema or {})
                inject = (
                    "Respond ONLY with a JSON object that strictly conforms to this JSON Schema:\n"
                    f"{schema_str}"
                )
            if inject:
                if system_value:
                    system_value = f"{system_value}\n\n{inject}"
                else:
                    system_value = inject

        if system_value is not None:
            payload_data["system"] = system_value
        if request.stream is not None:
            payload_data["stream"] = request.stream

        # Tools mapping (OpenAI function tools -> Anthropic custom tools)
        tools_in = request.tools or []
        if tools_in:
            anth_tools: list[dict[str, Any]] = []
            for t in tools_in:
                if t.type == "function" and t.function is not None:
                    fn = t.function
                    anth_tools.append(
                        {
                            "type": "custom",
                            "name": fn.name,
                            "description": fn.description,
                            "input_schema": fn.parameters.model_dump()
                            if hasattr(fn.parameters, "model_dump")
                            else (fn.parameters or {}),
                        }
                    )
            if anth_tools:
                payload_data["tools"] = anth_tools

        # tool_choice mapping
        tool_choice = request.tool_choice
        parallel_tool_calls = request.parallel_tool_calls
        disable_parallel = None
        if isinstance(parallel_tool_calls, bool):
            disable_parallel = not parallel_tool_calls

        if tool_choice is not None:
            anth_choice: dict[str, Any] | None = None
            if isinstance(tool_choice, str):
                if tool_choice == "none":
                    anth_choice = {"type": "none"}
                elif tool_choice == "auto":
                    anth_choice = {"type": "auto"}
                elif tool_choice == "required":
                    anth_choice = {"type": "any"}
            elif isinstance(tool_choice, dict):
                # Handle dict input like {"type": "function", "function": {"name": "search"}}
                if tool_choice.get("type") == "function" and isinstance(
                    tool_choice.get("function"), dict
                ):
                    anth_choice = {
                        "type": "tool",
                        "name": tool_choice["function"].get("name"),
                    }
            elif hasattr(tool_choice, "type") and hasattr(tool_choice, "function"):
                # e.g., ChatCompletionNamedToolChoice pydantic model
                if tool_choice.type == "function" and tool_choice.function is not None:
                    anth_choice = {
                        "type": "tool",
                        "name": tool_choice.function.name,
                    }
            if anth_choice is not None:
                if disable_parallel is not None and anth_choice["type"] in {
                    "auto",
                    "any",
                    "tool",
                }:
                    anth_choice["disable_parallel_tool_use"] = disable_parallel
                payload_data["tool_choice"] = anth_choice

        # Thinking configuration
        thinking_cfg = self._derive_thinking_config(model, request)
        if thinking_cfg is not None:
            payload_data["thinking"] = thinking_cfg
            # Ensure token budget fits under max_tokens
            budget = thinking_cfg.get("budget_tokens", 0)
            if isinstance(budget, int) and max_tokens <= budget:
                payload_data["max_tokens"] = budget + 64
            # Temperature constraint when thinking enabled
            payload_data["temperature"] = 1.0

        # Validate against Anthropic model to ensure shape
        return CreateMessageRequest.model_validate(payload_data)

    def _derive_thinking_config(
        self, model: str, request: ChatCompletionRequest
    ) -> dict[str, Any] | None:
        """Derive Anthropic thinking config from OpenAI fields and model name.

        Rules:
        - If model matches o1/o3 families, enable thinking by default with model-specific budget
        - Map reasoning_effort: low=1000, medium=5000, high=10000
        - o3*: 10000; o1-mini: 3000; other o1*: 5000
        - If thinking is enabled, return {"type":"enabled","budget_tokens":N}
        - Otherwise return None
        """
        # Explicit reasoning_effort mapping
        effort = getattr(request, "reasoning_effort", None)
        effort = effort.strip().lower() if isinstance(effort, str) else ""
        effort_budgets = {"low": 1000, "medium": 5000, "high": 10000}

        budget: int | None = None
        if effort in effort_budgets:
            budget = effort_budgets[effort]

        m = model.lower()
        # Model defaults if budget not set by effort
        if budget is None:
            if m.startswith("o3"):
                budget = 10000
            elif m.startswith("o1-mini"):
                budget = 3000
            elif m.startswith("o1"):
                budget = 5000

        if budget is None:
            return None

        return {"type": "enabled", "budget_tokens": budget}
