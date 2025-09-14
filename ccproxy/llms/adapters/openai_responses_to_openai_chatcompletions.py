from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel, TypeAdapter

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseObject,
)


class OpenAIResponsesToOpenAIChatAdapter(
    BaseAPIAdapter[
        ChatCompletionRequest,
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
    def _dict_to_request_model(self, request: dict[str, Any]) -> openai_models.ChatCompletionRequest:
        return openai_models.ChatCompletionRequest.model_validate(request)

    def _dict_to_response_model(self, response: dict[str, Any]) -> ResponseObject:
        return ResponseObject.model_validate(response)

    def _dict_to_error_model(self, error: dict[str, Any]) -> BaseModel:
        return BaseModel.model_validate(error)

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
        """Convert request using typed models - delegate to OpenAI Chat to Responses adapter."""
        from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
            OpenAIChatToOpenAIResponsesAdapter,
        )

        # Use the dedicated adapter for the reverse transformation
        adapter = OpenAIChatToOpenAIResponsesAdapter()
        return await adapter.adapt_request_typed(request)

    async def adapt_response_typed(self, response: BaseModel) -> ChatCompletionResponse:
        """Convert ResponseObject to ChatCompletionResponse using typed models."""
        if not isinstance(response, ResponseObject):
            raise ValueError(f"Expected ResponseObject, got {type(response)}")

        return await self._convert_response_typed(response)

    async def _convert_response_typed(
        self, response: ResponseObject
    ) -> ChatCompletionResponse:
        """Convert ResponseObject to ChatCompletionResponse using typed models."""
        # Find first message output and aggregate output_text parts
        text = ""
        for item in response.output or []:
            if hasattr(item, "type") and item.type == "message":
                parts: list[str] = []
                for part in getattr(item, "content", []):
                    if hasattr(part, "type") and part.type == "output_text":
                        if hasattr(part, "text") and isinstance(part.text, str):
                            parts.append(part.text)
                text = "".join(parts)
                break

        # Create usage object
        usage = None
        if response.usage:
            usage = openai_models.CompletionUsage(
                prompt_tokens=response.usage.input_tokens or 0,
                completion_tokens=response.usage.output_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        # Create the response
        return ChatCompletionResponse(
            id=response.id or "chatcmpl-resp",
            choices=[
                openai_models.Choice(
                    index=0,
                    message=openai_models.ResponseMessage(
                        role="assistant", content=text
                    ),
                    finish_reason="stop",
                )
            ],
            created=0,
            model=response.model or "",
            object="chat.completion",
            usage=usage
            or openai_models.CompletionUsage(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
        )

    def adapt_stream_typed(
        self, stream: AsyncIterator[openai_models.AnyStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Convert OpenAI Response stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream_typed(stream)

    async def _convert_stream_typed(
        self, stream: AsyncIterator[openai_models.AnyStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        model_id = ""
        async for event_wrapper in stream:
            if hasattr(event_wrapper, "root"):
                evt = event_wrapper.root
            else:
                evt = event_wrapper  # type: ignore
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
                if evt.response and evt.response.usage:  # type: ignore
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=evt.response.usage.input_tokens,  # type: ignore
                        completion_tokens=evt.response.usage.output_tokens,  # type: ignore
                        total_tokens=evt.response.usage.total_tokens,  # type: ignore
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
        """Convert error using typed models - passthrough for now."""
        return error

    async def adapt_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_request = self._dict_to_request_model(request)
        typed_result = await self.adapt_request_typed(typed_request)
        return typed_result.model_dump()

    async def adapt_response(self, response: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_response = self._dict_to_response_model(response)
        typed_result = await self.adapt_response_typed(typed_response)
        return typed_result.model_dump()

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Legacy dict interface - delegates to typed implementation."""
        typed_error = self._dict_to_error_model(error)
        typed_result = await self.adapt_error_typed(typed_error)
        return typed_result.model_dump()
