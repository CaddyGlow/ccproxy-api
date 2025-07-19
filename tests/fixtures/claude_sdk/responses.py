"""Response data for Claude SDK mocking.

This module contains standard response data used by Claude SDK mocks.
These responses match the format expected from the Claude SDK service.
"""

from typing import Any


# Standard completion response data
CLAUDE_SDK_STANDARD_COMPLETION = {
    "id": "msg_01234567890",
    "type": "message",
    "role": "assistant",
    "content": [{"type": "text", "text": "Hello! How can I help you?"}],
    "model": "claude-3-5-sonnet-20241022",
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 8},
}

# Model list response data
CLAUDE_SDK_MODEL_LIST = [
    {
        "id": "claude-3-5-sonnet-20241022",
        "object": "model",
        "created": 1677610602,
        "owned_by": "anthropic",
    },
    {
        "id": "claude-3-opus-20240229",
        "object": "model",
        "created": 1677610602,
        "owned_by": "anthropic",
    },
]

# Streaming events for SSE responses
CLAUDE_SDK_STREAMING_EVENTS = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 10, "output_tokens": 0},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hello"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": " world!"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
]

# Supported models for validation
SUPPORTED_CLAUDE_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-opus-20240229",
    "claude-3-haiku-20240307",
]


def get_claude_sdk_responses() -> dict[str, Any]:
    """Get all Claude SDK response data as a dictionary.

    Returns:
        Dictionary containing all standard Claude SDK responses
    """
    return {
        "standard_completion": CLAUDE_SDK_STANDARD_COMPLETION,
        "model_list": CLAUDE_SDK_MODEL_LIST,
        "streaming_events": CLAUDE_SDK_STREAMING_EVENTS,
        "supported_models": SUPPORTED_CLAUDE_MODELS,
    }
