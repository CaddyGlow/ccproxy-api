from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel, TypeAdapter

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import ANTHROPIC_TO_OPENAI_FINISH_REASON
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

    # Model conversion helpers
    def _dict_to_request_model(
        self, request: dict[str, Any]
    ) -> anthropic_models.CreateMessageRequest:
        """Convert dict to CreateMessageRequest."""
        # Preprocess tools to satisfy union discriminator if missing
        req_dict = dict(request)
        tools_in = req_dict.get("tools")
        if isinstance(tools_in, list):
            new_tools = []
            for t in tools_in:
                if isinstance(t, dict) and "type" not in t:
                    t = {"type": "custom", **t}
                new_tools.append(t)
            req_dict["tools"] = new_tools
        return anthropic_models.CreateMessageRequest.model_validate(req_dict)

    def _dict_to_response_model(
        self, response: dict[str, Any]
    ) -> anthropic_models.MessageResponse:
        """Convert dict to MessageResponse."""
        return anthropic_models.MessageResponse.model_validate(response)

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        """Convert dict to ErrorResponse."""
        return anthropic_models.ErrorResponse.model_validate(error)

    def _dict_stream_to_typed_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[anthropic_models.MessageStreamEvent]:
        """Convert dict stream to MessageStreamEvent stream."""

        event_adapter: TypeAdapter[anthropic_models.MessageStreamEvent] = TypeAdapter(
            anthropic_models.MessageStreamEvent
        )

        async def typed_generator() -> AsyncIterator[
            anthropic_models.MessageStreamEvent
        ]:
            async for chunk_dict in stream:
                try:
                    yield event_adapter.validate_python(chunk_dict)
                except Exception:
                    # Skip invalid chunks in stream
                    continue

        return typed_generator()

    # New strongly-typed methods
    async def adapt_request_typed(self, request: BaseModel) -> BaseModel:
        """Convert Anthropic CreateMessageRequest to OpenAI ChatCompletionRequest."""
        if not isinstance(request, anthropic_models.CreateMessageRequest):
            raise ValueError(f"Expected CreateMessageRequest, got {type(request)}")

        return await self._convert_request_typed(request)

    async def adapt_response_typed(self, response: BaseModel) -> BaseModel:
        """Convert Anthropic MessageResponse to OpenAI ChatCompletionResponse."""
        if not isinstance(response, anthropic_models.MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return await self._convert_response_typed(response)

    def adapt_stream_typed(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        """Convert Anthropic MessageStreamEvent stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream_typed(stream)

    async def _convert_stream_typed(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
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

    async def adapt_error_typed(self, error: BaseModel) -> BaseModel:
        """Convert error response - pass through for now."""
        return error

    # Implementation methods
    async def _convert_request_typed(
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

    async def _convert_response_typed(
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

        usage = response.usage
        prompt_tokens = int(usage.input_tokens or 0)
        completion_tokens = int(usage.output_tokens or 0)

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
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return openai_models.ChatCompletionResponse.model_validate(payload)

    async def _adapt_request_dict_impl(self, request: dict[str, Any]) -> dict[str, Any]:
        """Implementation moved from adapt_request - works with dicts."""
        # Preprocess tools to satisfy union discriminator if missing
        req_dict = dict(request)
        tools_in = req_dict.get("tools")
        if isinstance(tools_in, list):
            new_tools = []
            for t in tools_in:
                if isinstance(t, dict) and "type" not in t:
                    t = {"type": "custom", **t}
                new_tools.append(t)
            req_dict["tools"] = new_tools

        # Validate incoming as Anthropic CreateMessageRequest
        anthropic_request = anthropic_models.CreateMessageRequest.model_validate(
            req_dict
        )

        openai_messages: list[dict[str, Any]] = []
        # System prompt
        if anthropic_request.system:
            if isinstance(anthropic_request.system, str):
                sys_content = anthropic_request.system
            else:
                sys_content = "".join(block.text for block in anthropic_request.system)
            if sys_content:
                openai_messages.append({"role": "system", "content": sys_content})

        # User/assistant messages with text + data-url images
        for msg in anthropic_request.messages:
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
        if anthropic_request.tools:
            for tool in anthropic_request.tools:
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
            "model": anthropic_request.model,
            "messages": openai_messages,
            "max_completion_tokens": anthropic_request.max_tokens,
        }
        if tools:
            params["tools"] = tools

        # tool_choice mapping
        tc = anthropic_request.tool_choice
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
        req = openai_models.ChatCompletionRequest.model_validate(params)
        return req.model_dump()

    # Override to delegate to typed implementation
    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_request = self._dict_to_request_model(request)
        typed_result = await self.adapt_request_typed(typed_request)
        return typed_result.model_dump()

    async def _adapt_response_dict_impl(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Implementation moved from adapt_response - works with dicts."""
        content_blocks = response.get("content") or []
        parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif btype == "thinking":
                thinking = block.get("thinking")
                signature = block.get("signature")
                if isinstance(thinking, str):
                    sig_attr = (
                        f' signature="{signature}"'
                        if isinstance(signature, str) and signature
                        else ""
                    )
                    parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")

        content_text = "".join(parts)

        stop_reason = response.get("stop_reason")
        finish_reason = ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
            stop_reason or "end_turn", "stop"
        )

        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)

        payload = {
            "id": response.get("id"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content_text},
                    "finish_reason": finish_reason,
                }
            ],
            "created": 0,
            "model": response.get("model"),
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        return openai_models.ChatCompletionResponse.model_validate(payload).model_dump()

    # Override to delegate to typed implementation
    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_response = self._dict_to_response_model(response)
        typed_result = await self.adapt_response_typed(typed_response)
        return typed_result.model_dump()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        if error.get("type") == "error" and isinstance(error.get("error"), dict):
            anthropic_error = error["error"]
            return {
                "error": {
                    "message": anthropic_error.get("message", "Unknown error"),
                    "type": anthropic_error.get("type", "api_error"),
                    "param": None,
                    "code": None,
                }
            }
        return {
            "error": {
                "message": "Unknown or malformed Anthropic error",
                "type": "api_error",
                "param": None,
                "code": None,
            }
        }


# Backward-compatible alias for tests
class AnthropicToOpenAIChatCompletionsAdapter(AnthropicMessagesToOpenAIChatAdapter):
    pass
