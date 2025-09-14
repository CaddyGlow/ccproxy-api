from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai.models import ChatCompletionChunk, ChatCompletionResponse


class OpenAIResponsesToOpenAIChatAdapter(BaseAPIAdapter):
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

    def _dict_to_response_model(self, response: dict[str, Any]) -> BaseModel:
        return BaseModel(**response)  # Minimal implementation

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        return BaseModel(**error)  # Minimal implementation

    def _dict_stream_to_typed_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncIterator[BaseModel]:
        async def generator() -> AsyncIterator[BaseModel]:
            async for item in stream:
                yield BaseModel(**item)

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
        async def generator() -> AsyncGenerator[BaseModel, None]:
            async for item in stream:
                yield item  # Pass through for now

        return generator()

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

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        async def generator() -> AsyncGenerator[dict[str, Any], None]:
            model_id = ""
            async for evt in stream:
                if not isinstance(evt, dict):
                    continue
                etype = evt.get("type")
                if etype == "response.created":
                    model_id = (evt.get("response") or {}).get("model", "")
                elif etype == "response.output_text.delta":
                    delta_text = evt.get("delta") or ""
                    if delta_text:
                        chunk = {
                            "id": "chatcmpl-stream",
                            "object": "chat.completion.chunk",
                            "created": 0,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "content": delta_text,
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                        yield ChatCompletionChunk.model_validate(chunk).model_dump()
                elif etype in (
                    "response.completed",
                    "response.incomplete",
                    "response.failed",
                ):
                    usage_obj = (evt.get("response") or {}).get("usage") or {}
                    # Final chunk with finish reason
                    final = {
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": model_id,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop",
                            }
                        ],
                    }
                    # Optionally include usage if present
                    if usage_obj:
                        final["usage"] = {
                            "prompt_tokens": int(usage_obj.get("input_tokens") or 0),
                            "completion_tokens": int(
                                usage_obj.get("output_tokens") or 0
                            ),
                            "total_tokens": int(usage_obj.get("total_tokens") or 0),
                        }
                    yield ChatCompletionChunk.model_validate(final).model_dump()
                    break

        return generator()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        return error
