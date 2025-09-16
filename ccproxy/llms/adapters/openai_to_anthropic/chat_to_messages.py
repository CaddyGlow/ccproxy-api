from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Literal, cast

from pydantic import BaseModel

from ccproxy.llms.adapters.anthropic_to_openai.messages_to_chat import (
    convert_anthropic_message_to_chat_response,
)
from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.openai_to_anthropic.helpers import (
    convert__openai_to_anthropic__error,
)
from ccproxy.llms.adapters.shared import (
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
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
    ChatCompletionResponse,
)


FinishReason = Literal["stop", "length", "tool_calls"]


class OpenAIChatToAnthropicMessagesAdapter(
    BaseAPIAdapter[
        ChatCompletionRequest,
        ChatCompletionResponse,
        ChatCompletionResponse,
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
        return convert__openai_to_anthropic__error(error)
