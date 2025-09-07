"""Plugin-level conftest for Codex plugin tests.

Reuses shared high-performance integration fixtures and external API mocks.
"""

pytest_plugins = [
    "tests.fixtures.integration",
    "tests.fixtures.external_apis.openai_codex_api",
]


def pytest_configure(config):  # type: ignore[no-redef]
    """Register custom markers used by Codex plugin tests."""
    config.addinivalue_line("markers", "codex: mark test as codex plugin test")
    config.addinivalue_line("markers", "integration: integration tests")


# Local fallback for codex_responses to satisfy httpx mocks when root conftest
# is not auto-discovered in selective runs.
from typing import Any
import pytest


@pytest.fixture
def codex_responses() -> dict[str, Any]:
    return {
        "standard_completion": {
            "id": "codex_01234567890",
            "object": "codex.response",
            "created": 1234567890,
            "model": "gpt-5",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you with coding today?",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 12,
                "total_tokens": 22,
            },
        },
        "error_response": {
            "error": {
                "type": "invalid_request_error",
                "message": "Invalid model specified",
                "code": "invalid_model",
            }
        },
    }
