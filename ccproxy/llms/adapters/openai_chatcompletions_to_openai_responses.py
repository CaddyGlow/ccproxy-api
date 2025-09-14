from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai.models import ResponseRequest


class OpenAIChatToOpenAIResponsesAdapter(BaseAPIAdapter):
    """OpenAI Chat → OpenAI Responses request adapter (minimal).

    Implemented
    - model: passthrough
    - max_completion_tokens/max_tokens → `max_output_tokens`
    - messages: maps the last `user` message text to a single Responses `input` message

    TODO
    - Map all conversation turns to multi-item `input` if needed
    - Map richer contents (images, tools) to Responses-supported forms
    - Pass through response_format as-is if present on Chat (hybrid flows)
    """

    def __init__(self) -> None:
        super().__init__(name="openai_chat_to_openai_responses")

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        model = request.get("model")
        max_out = request.get("max_completion_tokens") or request.get("max_tokens")

        # Find the last user message
        user_text: str | None = None
        for msg in reversed(request.get("messages", []) or []):
            if msg.get("role") == "user":
                content = msg.get("content")
                if isinstance(content, list):
                    texts = [
                        part.get("text") for part in content if isinstance(part, dict)
                    ]
                    user_text = " ".join([t for t in texts if t])
                else:
                    user_text = content
                break

        input_msg = None
        if user_text:
            input_msg = {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text,
                    }
                ],
            }

        payload: dict[str, Any] = {
            "model": model,
        }
        if max_out is not None:
            payload["max_output_tokens"] = int(max_out)
        if input_msg:
            payload["input"] = [input_msg]

        # Structured outputs: map Chat response_format to Responses text.format
        resp_fmt = request.get("response_format")
        if isinstance(resp_fmt, dict):
            rftype = resp_fmt.get("type")
            if rftype == "json_object":
                payload["text"] = {"format": {"type": "json_object"}}
            elif rftype == "json_schema":
                js = resp_fmt.get("json_schema") or {}
                # Pass through name/schema/strict if provided
                fmt = {"type": "json_schema"}
                fmt.update(
                    {
                        k: v
                        for k, v in js.items()
                        if k in {"name", "schema", "strict", "$defs", "description"}
                    }
                )
                payload["text"] = {"format": fmt}

        return ResponseRequest.model_validate(payload).model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        # Delegate to Responses -> Chat adapter for converting results
        from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
            OpenAIResponsesToOpenAIChatAdapter,
        )

        return await OpenAIResponsesToOpenAIChatAdapter().adapt_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        # Delegate streaming conversion as well
        from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
            OpenAIResponsesToOpenAIChatAdapter,
        )

        return OpenAIResponsesToOpenAIChatAdapter().adapt_stream(stream)

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
