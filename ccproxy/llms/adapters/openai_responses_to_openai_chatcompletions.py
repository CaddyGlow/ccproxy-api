from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai.models import ChatCompletionResponse


class OpenAIResponsesToOpenAIChatAdapter(BaseAPIAdapter):
    """Map OpenAI Responses result to Chat Completions response."""

    def __init__(self) -> None:
        super().__init__(name="openai_responses_to_openai_chat")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Find first message output and aggregate output_text parts
        output = response.get("output") or []
        text = ""
        for item in output:
            if isinstance(item, dict) and item.get("type") == "message":
                content_list = item.get("content") or []
                parts: list[str] = []
                for part in content_list:
                    if isinstance(part, dict) and part.get("type") == "output_text":
                        t = part.get("text")
                        if isinstance(t, str):
                            parts.append(t)
                text = "".join(parts)
                break

        usage = response.get("usage") or {}
        payload = {
            "id": response.get("id"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
            "created": 0,
            "model": response.get("model"),
            "object": "chat.completion",
            "usage": {
                "prompt_tokens": int(usage.get("input_tokens") or 0),
                "completion_tokens": int(usage.get("output_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
        }
        return ChatCompletionResponse.model_validate(payload).model_dump()

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
