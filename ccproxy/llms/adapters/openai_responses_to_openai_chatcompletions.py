from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.mapping import safe_extract_usage_tokens
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

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        """Convert request using typed models - delegate to OpenAI Chat to Responses adapter."""
        from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
            OpenAIChatToOpenAIResponsesAdapter,
        )

        # Use the dedicated adapter for the reverse transformation
        adapter = OpenAIChatToOpenAIResponsesAdapter()
        return await adapter.adapt_request(request)

    async def adapt_response(self, response: BaseModel) -> ChatCompletionResponse:
        """Convert ResponseObject to ChatCompletionResponse using typed models."""
        if not isinstance(response, ResponseObject):
            raise ValueError(f"Expected ResponseObject, got {type(response)}")

        return await self._convert_response(response)

    async def _convert_response(
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

        # Create usage object with better token extraction
        usage = None
        if response.usage:
            input_tokens, output_tokens, _ = safe_extract_usage_tokens(response.usage)
            usage = openai_models.CompletionUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
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

    def adapt_stream(
        self, stream: AsyncIterator[openai_models.AnyStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Convert OpenAI Response stream to OpenAI ChatCompletionChunk stream."""
        return self._convert_stream(stream)

    async def _convert_stream(
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
                    input_tokens, output_tokens, _ = safe_extract_usage_tokens(
                        evt.response.usage
                    )  # type: ignore
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
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

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert error using typed models - normalize OpenAI error format."""
        from ccproxy.llms.adapters.mapping import normalize_openai_error

        return normalize_openai_error(error)
