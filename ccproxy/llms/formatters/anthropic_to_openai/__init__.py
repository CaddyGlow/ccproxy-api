"""Facade module exposing Anthropic→OpenAI formatter entry points."""

import sys
from types import ModuleType

from . import streams as _streams
from .errors import convert__anthropic_to_openai__error
from .requests import (
    convert__anthropic_message_to_openai_chat__request,
    convert__anthropic_message_to_openai_responses__request,
)
from .responses import (
    convert__anthropic_message_to_openai_chat__response,
    convert__anthropic_message_to_openai_responses__response,
    convert__anthropic_usage_to_openai_completion__usage,
    convert__anthropic_usage_to_openai_responses__usage,
)
from .streams import (
    AnthropicToOpenAIChatStreamAdapter,
    AnthropicToOpenAIResponsesStreamAdapter,
    convert__anthropic_message_to_openai_chat__stream,
    convert__anthropic_message_to_openai_responses__stream,
)


__all__ = [
    "convert__anthropic_to_openai__error",
    "convert__anthropic_message_to_openai_chat__request",
    "convert__anthropic_message_to_openai_responses__request",
    "convert__anthropic_message_to_openai_chat__response",
    "convert__anthropic_message_to_openai_responses__response",
    "convert__anthropic_usage_to_openai_completion__usage",
    "convert__anthropic_usage_to_openai_responses__usage",
    "AnthropicToOpenAIChatStreamAdapter",
    "AnthropicToOpenAIResponsesStreamAdapter",
    "convert__anthropic_message_to_openai_chat__stream",
    "convert__anthropic_message_to_openai_responses__stream",
]


class _AnthropicToOpenAIModule(ModuleType):
    _propagated_names = {
        "AnthropicToOpenAIChatStreamAdapter",
        "AnthropicToOpenAIResponsesStreamAdapter",
    }

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in self._propagated_names and hasattr(_streams, name):
            setattr(_streams, name, value)


_module = sys.modules[__name__]
if not isinstance(_module, _AnthropicToOpenAIModule):
    _module.__class__ = _AnthropicToOpenAIModule
