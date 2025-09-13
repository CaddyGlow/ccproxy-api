from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import ANTHROPIC_TO_OPENAI_FINISH_REASON
from ccproxy.llms.openai.models import ChatCompletionResponse


class AnthropicMessagesToOpenAIChatAdapter(BaseAPIAdapter):
    """Map Anthropic Messages responses to OpenAI Chat Completions responses."""

    def __init__(self) -> None:
        super().__init__(name="anthropic_messages_to_openai_chat")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Extract primary text from Anthropic content blocks
        content_blocks = response.get("content") or []
        text_parts: list[str] = []
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
        content_text = "".join(text_parts)

        stop_reason = response.get("stop_reason")
        finish_reason = ANTHROPIC_TO_OPENAI_FINISH_REASON.get(stop_reason or "end_turn", "stop")

        usage = response.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens") or 0)
        completion_tokens = int(usage.get("output_tokens") or 0)

        payload = {
            "id": response.get("id"),
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content_text,
                    },
                    "finish_reason": finish_reason,
                }
            ],
            "created": 0,  # We don't have a timestamp here; callers may override
            "model": response.get("model"),
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

        return ChatCompletionResponse.model_validate(payload).model_dump()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
