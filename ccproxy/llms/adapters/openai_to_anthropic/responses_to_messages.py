from __future__ import annotations

import json
import re
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.adapters.shared import (
    convert_openai_error_to_anthropic,
    convert_openai_response_usage_to_anthropic_usage,
)
from ccproxy.llms.adapters.shared.usage import (
    convert_anthropic_usage_to_openai_response_usage,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.anthropic.models import MessageResponse as AnthropicMessageResponse
from ccproxy.llms.openai import models as openai_models
from ccproxy.llms.openai.models import ResponseRequest


THINKING_PATTERN = re.compile(
    r"<thinking(?:\s+signature=\"([^\"]*)\")?>(.*?)</thinking>",
    re.DOTALL,
)


class OpenAIResponsesToAnthropicAdapter(
    BaseAPIAdapter[
        BaseModel,
        BaseModel,
        openai_models.AnyStreamEvent,
    ]
):
    """OpenAI Responses → Anthropic Messages adapter (non-streaming + streaming subset)."""

    def __init__(self) -> None:
        super().__init__(name="openai_responses_to_anthropic")

    async def adapt_request(self, request: BaseModel) -> BaseModel:
        if not isinstance(request, ResponseRequest):
            raise ValueError(f"Expected ResponseRequest, got {type(request)}")

        return convert_openai_response_request_to_anthropic(request)

    async def adapt_response(self, response: BaseModel) -> AnthropicMessageResponse:
        if not isinstance(response, openai_models.ResponseObject):
            raise ValueError(f"Expected ResponseObject, got {type(response)}")

        return convert_openai_response_to_anthropic_message(response)

    def adapt_stream(
        self,
        stream: AsyncIterator[openai_models.AnyStreamEvent],
    ) -> AsyncGenerator[anthropic_models.MessageStreamEvent, None]:
        return self._convert_stream(stream)

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        return convert_openai_error_to_anthropic(error)


__all__ = [
    "OpenAIResponsesToAnthropicAdapter",
    "convert_openai_response_to_anthropic_message",
]
