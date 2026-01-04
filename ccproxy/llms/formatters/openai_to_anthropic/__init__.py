"""Facade module exposing OpenAI→Anthropic formatter entry points."""

import sys
from types import ModuleType

from . import streams as _streams
from .errors import convert__openai_to_anthropic__error
from .requests import (
    convert__openai_chat_to_anthropic_message__request,
    convert__openai_responses_to_anthropic_message__request,
)
from .responses import (
    convert__openai_chat_to_anthropic_messages__response,
    convert__openai_responses_to_anthropic_message__response,
    convert__openai_responses_usage_to_anthropic__usage,
    convert__openai_responses_usage_to_openai_completion__usage,
)
from .streams import (
    OpenAIChatToAnthropicStreamAdapter,
    OpenAIResponsesToAnthropicStreamAdapter,
    convert__openai_chat_to_anthropic_messages__stream,
    convert__openai_responses_to_anthropic_messages__stream,
)


__all__ = [
    "convert__openai_to_anthropic__error",
    "convert__openai_chat_to_anthropic_message__request",
    "convert__openai_responses_to_anthropic_message__request",
    "convert__openai_chat_to_anthropic_messages__response",
    "convert__openai_responses_to_anthropic_message__response",
    "convert__openai_responses_usage_to_anthropic__usage",
    "convert__openai_responses_usage_to_openai_completion__usage",
    "OpenAIChatToAnthropicStreamAdapter",
    "OpenAIResponsesToAnthropicStreamAdapter",
    "convert__openai_chat_to_anthropic_messages__stream",
    "convert__openai_responses_to_anthropic_messages__stream",
]


class _OpenAIToAnthropicModule(ModuleType):
    _propagated_names = {
        "OpenAIChatToAnthropicStreamAdapter",
        "OpenAIResponsesToAnthropicStreamAdapter",
    }

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in self._propagated_names and hasattr(_streams, name):
            setattr(_streams, name, value)


_module = sys.modules[__name__]
if not isinstance(_module, _OpenAIToAnthropicModule):
    _module.__class__ = _OpenAIToAnthropicModule
