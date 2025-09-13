from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import DEFAULT_MAX_TOKENS
from ccproxy.llms.anthropic.models import CreateMessageRequest


class OpenAIChatToAnthropicMessagesAdapter(BaseAPIAdapter):
    """Map OpenAI Chat Completions requests to Anthropic Messages requests.

    This adapter focuses on a reliable core subset:
    - model: passthrough
    - messages: system -> `system`, user/assistant -> `messages`
    - max_completion_tokens/max_tokens: mapped to Anthropic `max_tokens`
    - stream: passthrough if present
    Other fields can be added incrementally as needed.
    """

    def __init__(self) -> None:
        super().__init__(name="openai_chat_to_anthropic_messages")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request.get("model")

        # Determine max tokens
        max_tokens = request.get("max_completion_tokens") or request.get("max_tokens")
        if max_tokens is None:
            max_tokens = DEFAULT_MAX_TOKENS

        # Extract system message if present
        system_value: str | None = None
        out_messages: list[dict[str, Any]] = []
        for msg in request.get("messages", []) or []:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                # Only support simple string content here
                if isinstance(content, str):
                    system_value = content
                else:
                    # Best effort: join text parts if array
                    if isinstance(content, list):
                        texts = [part.get("text") for part in content if isinstance(part, dict)]
                        system_value = " ".join([t for t in texts if t]) or None
            elif role in ("user", "assistant"):
                # Pass through as simple text messages
                if isinstance(content, list):
                    # Best effort: collect text fields
                    texts = [part.get("text") for part in content if isinstance(part, dict)]
                    content = " ".join([t for t in texts if t])
                out_messages.append({"role": role, "content": content})

        payload = {
            "model": model,
            "messages": out_messages,
            "max_tokens": max_tokens,
        }
        if system_value is not None:
            payload["system"] = system_value
        if "stream" in request:
            payload["stream"] = bool(request.get("stream"))

        # Validate against Anthropic model to ensure shape
        return CreateMessageRequest.model_validate(payload).model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        raise NotImplementedError

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
