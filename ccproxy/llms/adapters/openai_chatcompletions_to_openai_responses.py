from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai.models import (
    ChatCompletionRequest,
    ResponseObject,
    ResponseRequest,
)


class OpenAIChatToOpenAIResponsesAdapter(
    BaseAPIAdapter[
        ChatCompletionRequest,
        ResponseObject,
        BaseModel,
    ]
):
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

    # Typed methods - delegate to dict implementation for now
    async def adapt_request(self, request: BaseModel) -> BaseModel:
        if not isinstance(request, ChatCompletionRequest):
            raise ValueError(f"Expected ChatCompletionRequest, got {type(request)}")
        return await self._convert_request(request)

    async def adapt_response(self, response: BaseModel) -> BaseModel:
        # Delegate to Responses -> Chat adapter for converting results
        from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
            OpenAIResponsesToOpenAIChatAdapter,
        )

        reverse_adapter = OpenAIResponsesToOpenAIChatAdapter()
        return await reverse_adapter.adapt_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        # Delegate streaming conversion as well
        from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
            OpenAIResponsesToOpenAIChatAdapter,
        )

        return OpenAIResponsesToOpenAIChatAdapter().adapt_stream(stream)  # type: ignore[arg-type]

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        return error  # Pass through

    async def _convert_request(
        self, request: ChatCompletionRequest
    ) -> ResponseRequest:
        """Convert ChatCompletionRequest to ResponseRequest using typed models."""
        model = request.model
        max_out = request.max_completion_tokens or request.max_tokens

        # Find the last user message
        user_text: str | None = None
        for msg in reversed(request.messages or []):
            if msg.role == "user":
                content = msg.content
                if isinstance(content, list):
                    texts = [
                        part.text
                        for part in content
                        if hasattr(part, "type")
                        and part.type == "text"
                        and hasattr(part, "text")
                    ]
                    user_text = " ".join([t for t in texts if t])
                else:
                    user_text = content
                break

        input_data = []
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
            input_data = [input_msg]

        payload_data: dict[str, Any] = {
            "model": model,
        }
        if max_out is not None:
            payload_data["max_output_tokens"] = int(max_out)
        if input_data:
            payload_data["input"] = input_data

        # Structured outputs: map Chat response_format to Responses text.format
        resp_fmt = request.response_format
        if resp_fmt is not None:
            if resp_fmt.type == "text":
                payload_data["text"] = {"format": {"type": "text"}}
            elif resp_fmt.type == "json_object":
                payload_data["text"] = {"format": {"type": "json_object"}}
            elif resp_fmt.type == "json_schema" and hasattr(resp_fmt, "json_schema"):
                js = resp_fmt.json_schema
                # Pass through name/schema/strict if provided
                fmt = {"type": "json_schema"}
                if js is not None:
                    js_dict = js.model_dump() if hasattr(js, "model_dump") else js
                    if js_dict is not None:
                        fmt.update(
                            {
                                k: v
                                for k, v in js_dict.items()
                                if k
                                in {"name", "schema", "strict", "$defs", "description"}
                            }
                        )
                payload_data["text"] = {"format": fmt}

        if request.tools:
            payload_data["tools"] = [
                tool.model_dump() if hasattr(tool, "model_dump") else tool
                for tool in request.tools
            ]

        return ResponseRequest.model_validate(payload_data)
