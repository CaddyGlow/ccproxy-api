"""Facade module exposing OpenAI↔OpenAI formatter entry points."""

import sys
from types import ModuleType

from . import streams as _streams
from .requests import (
    convert__openai_chat_to_openai_responses__request,
    convert__openai_responses_to_openaichat__request,
)
from .responses import (
    convert__openai_chat_to_openai_responses__response,
    convert__openai_completion_usage_to_openai_responses__usage,
    convert__openai_responses_to_openai_chat__response,
    convert__openai_responses_usage_to_openai_completion__usage,
)
from .streams import (
    OpenAIChatToResponsesStreamAdapter,
    OpenAIResponsesToChatStreamAdapter,
    convert__openai_chat_to_openai_responses__stream,
    convert__openai_responses_to_openai_chat__stream,
)


__all__ = [
    "convert__openai_chat_to_openai_responses__request",
    "convert__openai_responses_to_openaichat__request",
    "convert__openai_chat_to_openai_responses__response",
    "convert__openai_completion_usage_to_openai_responses__usage",
    "convert__openai_responses_to_openai_chat__response",
    "convert__openai_responses_usage_to_openai_completion__usage",
    "OpenAIChatToResponsesStreamAdapter",
    "OpenAIResponsesToChatStreamAdapter",
    "convert__openai_chat_to_openai_responses__stream",
    "convert__openai_responses_to_openai_chat__stream",
]


class _OpenAIToOpenAIModule(ModuleType):
    _propagated_names = {
        "OpenAIChatToResponsesStreamAdapter",
        "OpenAIResponsesToChatStreamAdapter",
    }

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        if name in self._propagated_names and hasattr(_streams, name):
            setattr(_streams, name, value)


_module = sys.modules[__name__]
if not isinstance(_module, _OpenAIToOpenAIModule):
    _module.__class__ = _OpenAIToOpenAIModule
