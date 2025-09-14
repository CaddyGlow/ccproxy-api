from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ResponseObject,
)


class OpenAIResponsesToOpenAIChatAdapter(
    BaseAPIAdapter[
        BaseModel,
        ResponseObject,
        openai_models.AnyStreamEvent,
    ]
):
    """OpenAI Responses → OpenAI Chat (result) adapter.

    Implemented
    - Extracts first `message` output and concatenates `output_text` parts to Chat `content`
    - Maps usage (input/output/total tokens)

    TODO
    - Map structured content (function/tool events) to Chat tool_calls where present
    - Consider end-state fields (refusal, status variants) if needed
    """

    def __init__(self) -> None:
        super().__init__(name="openai_responses_to_openai_chat")

    # Minimal implementations for abstract methods - delegate to dict-based logic
    def _dict_to_request_model(self, request: dict[str, Any]) -> BaseModel:
        return BaseModel(**request)  # Minimal implementation

    def _dict_to_response_model(self, response: dict[str, Any]) -> ResponseObject:
        return ResponseObject.model_validate(response)

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        return BaseModel(**error)  # Minimal implementation

    def _dict_stream_to_typed_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[openai_models.AnyStreamEvent]:
        event_adapter = TypeAdapter(openai_models.AnyStreamEvent)

        async def generator() -> AsyncIterator[openai_models.AnyStreamEvent]:
            async for item in stream:
                try:
                    yield event_adapter.validate_python(item)
                except Exception:
                    continue

        return generator()

    async def adapt_request_typed(self, request: BaseModel) -> BaseModel:
        request_dict = (
            request.model_dump() if hasattr(request, "model_dump") else dict(request)
        )
        result_dict = await self.adapt_request(request_dict)
        return BaseModel(**result_dict)

    async def adapt_response_typed(self, response: BaseModel) -> BaseModel:
        response_dict = (
            response.model_dump() if hasattr(response, "model_dump") else dict(response)
        )
        result_dict = await self.adapt_response(response_dict)
        return ChatCompletionResponse.model_validate(result_dict)

    def adapt_stream_typed(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert OpenAI Response stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream_typed(stream)

    async def _convert_stream_typed(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        model_id = ""
        async for evt_wrapper in stream:
            if not hasattr(evt_wrapper, "root"):
                continue
            evt = evt_wrapper.root
            if not hasattr(evt, "type"):
                continue

            if evt.type == "response.created":
                model_id = evt.response.model or ""
            elif evt.type == "response.output_text.delta":
                delta_text = evt.delta or ""
                if delta_text:
                    yield ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=delta_text
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
            elif evt.type in (
                "response.completed",
                "response.incomplete",
                "response.failed",
            ):
                usage = None
                if evt.response and evt.response.usage:
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=evt.response.usage.input_tokens,
                        completion_tokens=evt.response.usage.output_tokens,
                        total_tokens=evt.response.usage.total_tokens,
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
                            finish_reason="stop",
                        )
                    ],
                    usage=usage,
                )
                break

    async def adapt_error_typed(self, error: BaseModel) -> BaseModel:
        error_dict = error.model_dump() if hasattr(error, "model_dump") else dict(error)
        result_dict = await self.adapt_error(error_dict)
        return BaseModel(**result_dict)

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        # Delegate Chat -> Responses request mapping to the dedicated adapter
        from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
            OpenAIChatToOpenAIResponsesAdapter,
        )

        return await OpenAIChatToOpenAIResponsesAdapter().adapt_request(request)

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

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
