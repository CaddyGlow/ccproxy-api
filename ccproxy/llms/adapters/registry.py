"""Central registry describing built-in LLM format adapters.

Provides a single source of truth for adapter metadata so that plugin factories
and tests can reason about available conversions without relying on ad-hoc
import ordering.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple, Type

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter


@dataclass(frozen=True)
class AdapterRegistration:
    """Metadata about a typed adapter implementation."""

    name: str
    from_format: str
    to_format: str
    adapter_cls: Type[BaseAPIAdapter]
    request_model: Type[BaseModel]
    response_model: Type[BaseModel]
    stream_model: Type[BaseModel]
    description: str
    priority: int = 100

    @property
    def key(self) -> Tuple[str, str]:
        return self.from_format, self.to_format


# Import lazily to avoid issues during module initialization
from ccproxy.llms.adapters.anthropic_to_openai.messages_to_chat import (  # noqa: E402
    AnthropicMessagesToOpenAIChatAdapter,
)
from ccproxy.llms.adapters.anthropic_to_openai.messages_to_responses import (  # noqa: E402
    AnthropicMessagesToOpenAIResponsesAdapter,
)
from ccproxy.llms.adapters.openai_to_anthropic.chat_to_messages import (  # noqa: E402
    OpenAIChatToAnthropicMessagesAdapter,
)
from ccproxy.llms.adapters.openai_to_openai.chat_to_responses import (  # noqa: E402
    OpenAIChatToOpenAIResponsesAdapter,
)
from ccproxy.llms.adapters.openai_to_anthropic.responses_request_to_messages import (  # noqa: E402
    OpenAIResponsesRequestToAnthropicMessagesAdapter,
)
from ccproxy.llms.adapters.openai_to_anthropic.responses_to_messages import (  # noqa: E402
    OpenAIResponsesToAnthropicAdapter,
)
from ccproxy.llms.adapters.openai_to_openai.responses_to_chat import (  # noqa: E402
    OpenAIResponsesToOpenAIChatAdapter,
)
from ccproxy.llms.adapters.openai_to_openai.response_api_to_chat import (  # noqa: E402
    ResponseAPIToOpenAIChatAdapter,
)
from ccproxy.llms.anthropic import models as anthropic_models  # noqa: E402
from ccproxy.llms.openai import models as openai_models  # noqa: E402


_REGISTERED_ADAPTERS: tuple[AdapterRegistration, ...] = (
    AdapterRegistration(
        name="openai_chat_to_anthropic_messages",
        from_format="openai.chat_completions",
        to_format="anthropic.messages",
        adapter_cls=OpenAIChatToAnthropicMessagesAdapter,
        request_model=openai_models.ChatCompletionRequest,
        response_model=anthropic_models.MessageResponse,
        stream_model=anthropic_models.MessageStreamEvent,
        description="OpenAI ChatCompletions → Anthropic Messages",
    ),
    AdapterRegistration(
        name="anthropic_messages_to_openai_chat",
        from_format="anthropic.messages",
        to_format="openai.chat_completions",
        adapter_cls=AnthropicMessagesToOpenAIChatAdapter,
        request_model=anthropic_models.CreateMessageRequest,
        response_model=anthropic_models.MessageResponse,
        stream_model=anthropic_models.MessageStreamEvent,
        description="Anthropic Messages → OpenAI ChatCompletions",
    ),
    AdapterRegistration(
        name="anthropic_messages_to_openai_responses",
        from_format="anthropic.messages",
        to_format="openai.responses",
        adapter_cls=AnthropicMessagesToOpenAIResponsesAdapter,
        request_model=anthropic_models.CreateMessageRequest,
        response_model=anthropic_models.MessageResponse,
        stream_model=anthropic_models.MessageStreamEvent,
        description="Anthropic Messages → OpenAI Responses",
    ),
    AdapterRegistration(
        name="openai_responses_request_to_anthropic_messages",
        from_format="openai.responses.request",
        to_format="anthropic.messages",
        adapter_cls=OpenAIResponsesRequestToAnthropicMessagesAdapter,
        request_model=openai_models.ResponseRequest,
        response_model=anthropic_models.MessageResponse,
        stream_model=anthropic_models.MessageStreamEvent,
        description="OpenAI Responses request → Anthropic Messages",
    ),
    AdapterRegistration(
        name="openai_responses_to_anthropic",
        from_format="openai.responses",
        to_format="anthropic.messages",
        adapter_cls=OpenAIResponsesToAnthropicAdapter,
        request_model=openai_models.ResponseRequest,
        response_model=openai_models.ResponseObject,
        stream_model=openai_models.AnyStreamEvent,
        description="OpenAI Responses → Anthropic Messages",
    ),
    AdapterRegistration(
        name="openai_responses_to_openai_chat",
        from_format="openai.responses",
        to_format="openai.chat_completions",
        adapter_cls=OpenAIResponsesToOpenAIChatAdapter,
        request_model=openai_models.ChatCompletionRequest,
        response_model=openai_models.ResponseObject,
        stream_model=openai_models.AnyStreamEvent,
        description="OpenAI Responses → OpenAI ChatCompletions",
    ),
    AdapterRegistration(
        name="openai_chat_to_openai_responses",
        from_format="openai.chat_completions",
        to_format="openai.responses",
        adapter_cls=OpenAIChatToOpenAIResponsesAdapter,
        request_model=openai_models.ChatCompletionRequest,
        response_model=openai_models.ResponseObject,
        stream_model=openai_models.AnyStreamEvent,
        description="OpenAI ChatCompletions → OpenAI Responses",
    ),
    AdapterRegistration(
        name="response_api_to_openai_chat",
        from_format="openai.response_api",
        to_format="openai.chat_completions",
        adapter_cls=ResponseAPIToOpenAIChatAdapter,
        request_model=openai_models.ResponseRequest,
        response_model=openai_models.ChatCompletionResponse,
        stream_model=openai_models.AnyStreamEvent,
        description="Response API payloads → OpenAI ChatCompletions",
    ),
)


def iter_adapter_registrations() -> Iterable[AdapterRegistration]:
    """Iterate over the built-in adapter registrations in canonical order."""
    return iter(_REGISTERED_ADAPTERS)


def get_registered_adapter_map() -> dict[tuple[str, str], Type[BaseAPIAdapter]]:
    """Return mapping of format pair to adapter class for quick lookup."""
    return {reg.key: reg.adapter_cls for reg in _REGISTERED_ADAPTERS}


__all__ = [
    "AdapterRegistration",
    "iter_adapter_registrations",
    "get_registered_adapter_map",
]
