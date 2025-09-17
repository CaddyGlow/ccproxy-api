"""Shared test data constants and builders for API tests.

This module provides centralized test data to reduce duplication across test files
and ensure consistency in test scenarios.
"""

from typing import Any

from ccproxy.core.constants import (
    FORMAT_ANTHROPIC_MESSAGES,
    FORMAT_OPENAI_CHAT,
    FORMAT_OPENAI_RESPONSES,
)


# Standard model names used across tests
CLAUDE_SONNET_MODEL = "claude-3-5-sonnet-20241022"
INVALID_MODEL_NAME = "invalid-model"


def normalize_format(format_type: str) -> str:
    """Map canonical format identifiers to simplified categories for tests."""

    alias_map = {
        FORMAT_OPENAI_CHAT: "openai",
        FORMAT_OPENAI_RESPONSES: "response_api",
        FORMAT_ANTHROPIC_MESSAGES: "anthropic",
        "openai": "openai",
        "response_api": "response_api",
        "anthropic": "anthropic",
        "codex": "codex",
    }
    return alias_map.get(format_type, format_type)


# Common request data structures
STANDARD_OPENAI_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "messages": [{"role": "user", "content": "Hello, world!"}],
    "max_tokens": 100,
    "temperature": 0.7,
}

STANDARD_ANTHROPIC_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello, Claude!"}],
}

OPENAI_REQUEST_WITH_SYSTEM: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ],
    "max_tokens": 50,
}

ANTHROPIC_REQUEST_WITH_SYSTEM: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "max_tokens": 100,
    "system": "You are a helpful assistant.",
    "messages": [{"role": "user", "content": "Hello!"}],
}

# Error test cases
INVALID_MODEL_OPENAI_REQUEST: dict[str, Any] = {
    "model": INVALID_MODEL_NAME,
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50,
}

INVALID_MODEL_ANTHROPIC_REQUEST: dict[str, Any] = {
    "model": INVALID_MODEL_NAME,
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello"}],
}

MISSING_MESSAGES_OPENAI_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "max_tokens": 50,
}

EMPTY_MESSAGES_OPENAI_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "messages": [],
    "max_tokens": 50,
}

MALFORMED_MESSAGE_OPENAI_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "messages": [{"invalid_field": "user", "content": "Hello"}],
    "max_tokens": 50,
}

MISSING_MAX_TOKENS_ANTHROPIC_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
}

INVALID_ROLE_ANTHROPIC_REQUEST: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "max_tokens": 100,
    "messages": [{"role": "invalid", "content": "Hello"}],
}

# Streaming request variants
STREAMING_OPENAI_REQUEST: dict[str, Any] = {
    **STANDARD_OPENAI_REQUEST,
    "stream": True,
}

STREAMING_ANTHROPIC_REQUEST: dict[str, Any] = {
    **STANDARD_ANTHROPIC_REQUEST,
    "stream": True,
}

# Large request for testing body size limits
LARGE_CONTENT = "x" * 1000000  # 1MB of text
LARGE_REQUEST_ANTHROPIC: dict[str, Any] = {
    "model": CLAUDE_SONNET_MODEL,
    "max_tokens": 50,
    "messages": [{"role": "user", "content": LARGE_CONTENT}],
}

# Authentication test data
TEST_AUTH_TOKEN = "test-token-12345"
INVALID_AUTH_TOKEN = "invalid-token"

# OpenAI Codex request data structures
STANDARD_CODEX_REQUEST: dict[str, Any] = {
    "input": [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello, Codex!"}],
        }
    ],
    "model": "gpt-5",
    "store": False,
}

CODEX_REQUEST_WITH_SESSION: dict[str, Any] = {
    **STANDARD_CODEX_REQUEST,
    "session_id": "test-session-123",
}

# Codex streaming request variants
STREAMING_CODEX_REQUEST: dict[str, Any] = {
    **STANDARD_CODEX_REQUEST,
    "stream": True,
}

# Codex error test cases
INVALID_MODEL_CODEX_REQUEST: dict[str, Any] = {
    **STANDARD_CODEX_REQUEST,
    "model": INVALID_MODEL_NAME,
}

MISSING_INPUT_CODEX_REQUEST: dict[str, Any] = {
    "model": "gpt-5",
    "store": False,
    # Missing "input" field
}

EMPTY_INPUT_CODEX_REQUEST: dict[str, Any] = {
    "input": [],
    "model": "gpt-5",
    "store": False,
}

MALFORMED_INPUT_CODEX_REQUEST: dict[str, Any] = {
    "input": [
        {
            "invalid_field": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        }
    ],
    "model": "gpt-5",
    "store": False,
}

# Expected response field sets for validation
OPENAI_RESPONSE_FIELDS = {"id", "object", "created", "model", "choices", "usage"}
ANTHROPIC_RESPONSE_FIELDS = {
    "id",
    "type",
    "role",
    "content",
    "model",
    "stop_reason",
    "usage",
}
CODEX_RESPONSE_FIELDS = {
    "id",
    "object",
    "created",
    "model",
    "choices",
    "usage",
}

# E2E Endpoint Test Data
E2E_ENDPOINT_CONFIGURATIONS = [
    {
        "name": "copilot_chat_completions_stream",
        "endpoint": "/copilot/v1/chat/completions",
        "stream": True,
        "model": "gpt-4o",
        "format": FORMAT_OPENAI_CHAT,
        "description": "Copilot chat completions streaming",
    },
    {
        "name": "copilot_chat_completions",
        "endpoint": "/copilot/v1/chat/completions",
        "stream": False,
        "model": "gpt-4o",
        "format": FORMAT_OPENAI_CHAT,
        "description": "Copilot chat completions non-streaming",
    },
    {
        "name": "copilot_responses_stream",
        "endpoint": "/copilot/v1/responses",
        "stream": True,
        "model": "gpt-4o",
        "format": FORMAT_OPENAI_RESPONSES,
        "description": "Copilot responses streaming",
    },
    {
        "name": "copilot_responses",
        "endpoint": "/copilot/v1/responses",
        "stream": False,
        "model": "gpt-4o",
        "format": FORMAT_OPENAI_RESPONSES,
        "description": "Copilot responses non-streaming",
    },
    {
        "name": "anthropic_api_openai_stream",
        "endpoint": "/api/v1/chat/completions",
        "stream": True,
        "model": "claude-sonnet-4-20250514",
        "format": FORMAT_OPENAI_CHAT,
        "description": "Claude API OpenAI format streaming",
    },
    {
        "name": "anthropic_api_openai",
        "endpoint": "/api/v1/chat/completions",
        "stream": False,
        "model": "claude-sonnet-4-20250514",
        "format": FORMAT_OPENAI_CHAT,
        "description": "Claude API OpenAI format non-streaming",
    },
    {
        "name": "anthropic_api_responses_stream",
        "endpoint": "/api/v1/responses",
        "stream": True,
        "model": "claude-sonnet-4-20250514",
        "format": FORMAT_OPENAI_RESPONSES,
        "description": "Claude API Response format streaming",
    },
    {
        "name": "anthropic_api_responses",
        "endpoint": "/api/v1/responses",
        "stream": False,
        "model": "claude-sonnet-4-20250514",
        "format": FORMAT_OPENAI_RESPONSES,
        "description": "Claude API Response format non-streaming",
    },
    {
        "name": "codex_chat_completions_stream",
        "endpoint": "/api/codex/v1/chat/completions",
        "stream": True,
        "model": "gpt-5",
        "format": "openai",
        "description": "Codex chat completions streaming",
    },
    {
        "name": "codex_chat_completions",
        "endpoint": "/api/codex/v1/chat/completions",
        "stream": False,
        "model": "gpt-5",
        "format": "openai",
        "description": "Codex chat completions non-streaming",
    },
]


def create_openai_request(
    content: str = "Hello",
    model: str = CLAUDE_SONNET_MODEL,
    max_tokens: int = 50,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a customizable OpenAI request."""
    request = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
    }
    request.update(kwargs)
    return request


def create_anthropic_request(
    content: str = "Hello",
    model: str = CLAUDE_SONNET_MODEL,
    max_tokens: int = 50,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a customizable Anthropic request."""
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": content}],
    }
    request.update(kwargs)
    return request


def create_codex_request(
    content: str = "Hello",
    model: str = "gpt-5",
    store: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a customizable Codex request."""
    request = {
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }
        ],
        "model": model,
        "store": store,
    }
    request.update(kwargs)
    return request


def create_response_api_request(
    content: str = "Hello",
    model: str = CLAUDE_SONNET_MODEL,
    max_completion_tokens: int = 1000,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a customizable Response API request."""
    request = {
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": content}],
            }
        ],
    }
    request.update(kwargs)
    return request


def create_e2e_request_for_format(
    format_type: str,
    model: str,
    content: str = "Hello",
    stream: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Create a request for E2E testing based on format type."""
    normalized = normalize_format(format_type)

    if normalized == "openai":
        return create_openai_request(
            content=content,
            model=model,
            stream=stream,
            **kwargs,
        )
    elif normalized == "anthropic":
        return create_anthropic_request(
            content=content,
            model=model,
            stream=stream,
            **kwargs,
        )
    elif normalized == "response_api":
        return create_response_api_request(
            content=content,
            model=model,
            stream=stream,
            **kwargs,
        )
    elif normalized == "codex":
        return create_codex_request(
            content=content,
            model=model,
            stream=stream,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown format type: {format_type}")


def get_expected_response_fields(format_type: str) -> set[str]:
    """Get expected response fields for a given format type."""
    normalized = normalize_format(format_type)
    field_map = {
        "openai": OPENAI_RESPONSE_FIELDS,
        "anthropic": ANTHROPIC_RESPONSE_FIELDS,
        "response_api": CODEX_RESPONSE_FIELDS,  # Similar structure to OpenAI
        "codex": CODEX_RESPONSE_FIELDS,
    }
    return field_map.get(normalized, set())
