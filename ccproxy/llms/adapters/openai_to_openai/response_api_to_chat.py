from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shared import safe_extract_usage_tokens
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import (
    AnyStreamEvent,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    InputTokensDetails,
    MessageOutput,
    OutputTextContent,
    OutputTokensDetails,
    ResponseObject,
    ResponseRequest,
    ResponseUsage,
)


class ResponseAPIToOpenAIChatAdapter(
    BaseAPIAdapter[
        openai_models.ResponseRequest,
        openai_models.ChatCompletionResponse,
        openai_models.AnyStreamEvent,
    ]
):
    """Convert Response API payloads to OpenAI Chat Completions."""

    def __init__(self) -> None:
        super().__init__(name="response_api_to_openai_chat")

    async def adapt_request(self, request: BaseModel) -> ChatCompletionRequest:
        if not isinstance(request, ResponseRequest):
            raise ValueError(f"Expected ResponseRequest, got {type(request)}")

        return await self._convert_request(request)

    async def adapt_response(self, response: BaseModel) -> ResponseObject:
        if not isinstance(response, ChatCompletionResponse):
            raise ValueError(f"Expected ChatCompletionResponse, got {type(response)}")

        return await self._convert_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[AnyStreamEvent]
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        """Convert Response API stream events to OpenAI ChatCompletion chunks."""
        return response_stream_to_chat_chunks(stream)

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        # Chat Completions and Response API currently share the same error envelope
        return error

    async def _convert_request(self, request: ResponseRequest) -> ChatCompletionRequest:
        system_message: str | None = request.instructions
        messages: list[dict[str, Any]] = []

        # Handle string input shortcut
        if isinstance(request.input, str):
            messages.append({"role": "user", "content": request.input})
        else:
            for item in request.input or []:
                role = getattr(item, "role", None) or "user"
                content_blocks = getattr(item, "content", [])
                text_parts: list[str] = []

                for part in content_blocks or []:
                    if isinstance(part, dict):
                        if part.get("type") in {"input_text", "text"}:
                            text = part.get("text")
                            if isinstance(text, str):
                                text_parts.append(text)
                    else:
                        part_type = getattr(part, "type", None)
                        if part_type in {"input_text", "text"} and hasattr(
                            part, "text"
                        ):
                            text_value = part.text
                            if isinstance(text_value, str):
                                text_parts.append(text_value)

                if role == "system":
                    # Merge all system content into a single system message
                    system_message = " ".join([p for p in text_parts if p])
                else:
                    messages.append(
                        {
                            "role": role,
                            "content": " ".join([p for p in text_parts if p]) or None,
                        }
                    )

        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        # Provide a default user prompt if none extracted
        if not messages:
            messages.append({"role": "user", "content": ""})

        payload: dict[str, Any] = {
            "model": request.model or "gpt-4o-mini",
            "messages": messages,
        }

        if request.max_output_tokens is not None:
            payload["max_completion_tokens"] = request.max_output_tokens

        if request.stream is not None:
            payload["stream"] = request.stream

        if request.temperature is not None:
            payload["temperature"] = request.temperature

        if request.top_p is not None:
            payload["top_p"] = request.top_p

        if request.tools:
            payload["tools"] = request.tools

        if request.tool_choice is not None:
            payload["tool_choice"] = request.tool_choice

        if request.parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = request.parallel_tool_calls

        return ChatCompletionRequest.model_validate(payload)

    async def _convert_response(
        self, chat_response: ChatCompletionResponse
    ) -> ResponseObject:
        content_text = ""
        if chat_response.choices:
            first_choice = chat_response.choices[0]
            if first_choice.message and first_choice.message.content:
                content_text = first_choice.message.content

        message_output = MessageOutput(
            type="message",
            role="assistant",
            id=f"msg_{chat_response.id or 'unknown'}",
            status="completed",
            content=[OutputTextContent(type="output_text", text=content_text)],
        )

        usage: ResponseUsage | None = None
        if chat_response.usage:
            usage = ResponseUsage(
                input_tokens=chat_response.usage.prompt_tokens or 0,
                input_tokens_details=InputTokensDetails(cached_tokens=0),
                output_tokens=chat_response.usage.completion_tokens or 0,
                output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
                total_tokens=chat_response.usage.total_tokens or 0,
            )

        return ResponseObject(
            id=chat_response.id or "resp-unknown",
            object="response",
            created_at=int(time.time()),
            model=chat_response.model or "",
            status="completed",
            output=[message_output],
            parallel_tool_calls=False,
            usage=usage,
        )



def response_stream_to_chat_chunks(
    stream: AsyncIterator[AnyStreamEvent],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Convert Response API stream events to ChatCompletionChunk events."""

    async def generator() -> AsyncGenerator[ChatCompletionChunk, None]:
        model_id = ""
        async for event_wrapper in stream:
            if hasattr(event_wrapper, "root"):
                evt = event_wrapper.root
            else:
                evt = event_wrapper  # type: ignore[arg-type]
            if not hasattr(evt, "type"):
                continue

            if evt.type == "response.created":
                model_id = getattr(evt.response, "model", "")
            elif evt.type == "response.output_text.delta":
                delta = getattr(evt, "delta", None) or ""
                if delta:
                    yield ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=delta
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
            elif evt.type in {"response.completed", "response.incomplete", "response.failed"}:
                usage = None
                response_obj = getattr(evt, "response", None)
                if response_obj and getattr(response_obj, "usage", None):
                    input_tokens, output_tokens, _ = safe_extract_usage_tokens(
                        response_obj.usage
                    )
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

    return generator()
