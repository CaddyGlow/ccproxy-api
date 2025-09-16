from __future__ import annotations

import time
from collections.abc import AsyncGenerator, AsyncIterator

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.openai.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ResponseObject,
    ResponseRequest,
    ResponseUsage,
)


class ResponseAPIOpenAIBidirectionalAdapter(
    BaseAPIAdapter[
        BaseModel,
        BaseModel,
        BaseModel,
    ]
):
    """Bidirectional adapter for Response API ↔ OpenAI ChatCompletions.

    Handles both directions:
    - Request: ResponseRequest ↔ ChatCompletionRequest
    - Response: ResponseObject ↔ ChatCompletionResponse
    """

    def __init__(self) -> None:
        super().__init__(name="response_api_openai_bidirectional")

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        """Convert request - detects input type and converts appropriately."""
        if isinstance(request, ResponseRequest):
            # ResponseRequest → ChatCompletionRequest
            return await self._convert_response_request_to_chat(request)
        elif isinstance(request, ChatCompletionRequest):
            # ChatCompletionRequest → ResponseRequest
            return await self._convert_chat_request_to_response(request)
        else:
            raise ValueError(f"Unsupported request type: {type(request)}")

    async def adapt_response(self, response: BaseModel) -> BaseModel:
        """Convert response - detects input type and converts appropriately."""
        if isinstance(response, ResponseObject):
            # ResponseObject → ChatCompletionResponse
            return await self._convert_response_to_chat(response)
        elif isinstance(response, ChatCompletionResponse):
            # ChatCompletionResponse → ResponseObject
            return await self._convert_chat_to_response(response)
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert error - pass through for now."""
        return error

    async def _convert_response_request_to_chat(
        self, request: ResponseRequest
    ) -> ChatCompletionRequest:
        """Convert ResponseRequest to ChatCompletionRequest."""
        from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
            OpenAIChatToOpenAIResponsesAdapter,
        )

        # Delegate to existing adapter
        adapter = OpenAIChatToOpenAIResponsesAdapter()
        # This adapter does the reverse direction, so we need to create a ChatCompletionRequest
        # that would result in the given ResponseRequest when converted

        # For now, create a basic ChatCompletion request from the ResponseRequest
        messages = []
        if request.input:
            for item in request.input:
                if hasattr(item, "role") and hasattr(item, "content"):
                    content_text = ""
                    if hasattr(item, "content") and item.content:
                        for part in item.content:
                            if hasattr(part, "text") and hasattr(part, "type"):
                                if part.type in ["input_text", "text"]:
                                    content_text += part.text or ""

                    messages.append(
                        {
                            "role": item.role or "user",
                            "content": content_text,
                        }
                    )

        return ChatCompletionRequest(
            model=request.model or "gpt-4",
            messages=messages or [{"role": "user", "content": "Hello"}],
            max_tokens=getattr(request, "max_output_tokens", None),
        )

    async def _convert_chat_request_to_response(
        self, request: ChatCompletionRequest
    ) -> ResponseRequest:
        """Convert ChatCompletionRequest to ResponseRequest."""
        from ccproxy.llms.adapters.openai_chatcompletions_to_openai_responses import (
            OpenAIChatToOpenAIResponsesAdapter,
        )

        # Delegate to existing adapter
        adapter = OpenAIChatToOpenAIResponsesAdapter()
        return await adapter.adapt_request(request)

    async def _convert_response_to_chat(
        self, response: ResponseObject
    ) -> ChatCompletionResponse:
        """Convert ResponseObject to ChatCompletionResponse."""
        from ccproxy.llms.adapters.openai_responses_to_openai_chatcompletions import (
            OpenAIResponsesToOpenAIChatAdapter,
        )

        # Delegate to existing adapter
        adapter = OpenAIResponsesToOpenAIChatAdapter()
        return await adapter.adapt_response(response)

    async def _convert_chat_to_response(
        self, chat_response: ChatCompletionResponse
    ) -> ResponseObject:
        """Convert ChatCompletionResponse to ResponseObject."""
        # Extract content from the first choice
        content = ""
        if chat_response.choices and len(chat_response.choices) > 0:
            choice = chat_response.choices[0]
            if choice.message and choice.message.content:
                content = choice.message.content

        # Create output message
        output = [
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg_{chat_response.id}",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": content,
                    }
                ],
            }
        ]

        # Convert usage information
        usage = None
        if chat_response.usage:
            usage = ResponseUsage(
                input_tokens=chat_response.usage.prompt_tokens or 0,
                input_tokens_details={"cache_creation": 0, "cache_read": 0},
                output_tokens=chat_response.usage.completion_tokens or 0,
                output_tokens_details={"reasoning": 0},
                total_tokens=chat_response.usage.total_tokens or 0,
            )

        # Create the response object
        return ResponseObject(
            id=chat_response.id or "resp-unknown",
            object="response",
            created_at=int(time.time()),
            model=chat_response.model or "",
            output=output,
            status="completed",
            parallel_tool_calls=False,
            usage=usage
            or ResponseUsage(
                input_tokens=0,
                input_tokens_details={"cache_creation": 0, "cache_read": 0},
                output_tokens=0,
                output_tokens_details={"reasoning": 0},
                total_tokens=0,
            ),
        )

    def adapt_stream(
        self, stream: AsyncIterator[BaseModel]
    ) -> AsyncGenerator[BaseModel, None]:
        """Convert stream - for now, pass through."""
        return stream  # type: ignore
