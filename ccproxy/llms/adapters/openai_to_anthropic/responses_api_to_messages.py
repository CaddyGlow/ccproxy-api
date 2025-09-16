"""Compatibility exports for OpenAI → Anthropic response adapters."""

from __future__ import annotations

from ccproxy.llms.adapters.openai_to_anthropic.responses_request_to_messages import (
    OpenAIResponsesRequestToAnthropicMessagesAdapter,
    convert_anthropic_message_to_response_object,
    convert_openai_response_request_to_anthropic,
    derive_thinking_config,
)
from ccproxy.llms.adapters.openai_to_anthropic.responses_to_messages import (
    OpenAIResponsesToAnthropicAdapter,
    convert_openai_response_to_anthropic_message,
)

__all__ = [
    "OpenAIResponsesRequestToAnthropicMessagesAdapter",
    "OpenAIResponsesToAnthropicAdapter",
    "convert_openai_response_request_to_anthropic",
    "convert_openai_response_to_anthropic_message",
    "convert_anthropic_message_to_response_object",
    "derive_thinking_config",
]
