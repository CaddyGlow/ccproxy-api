"""Unit tests for Claude API request transformer."""

import json
from unittest.mock import Mock

import pytest

from ccproxy.plugins.claude_api.detection_service import ClaudeAPIDetectionService
from ccproxy.plugins.claude_api.models import (
    ClaudeCacheData,
    ClaudeCodeHeaders,
    SystemPromptData,
)
from ccproxy.plugins.claude_api.transformers.request import ClaudeAPIRequestTransformer


class TestClaudeAPIRequestTransformer:
    """Test suite for ClaudeAPIRequestTransformer."""

    def test_init_default_mode(self) -> None:
        """Test transformer initialization with default mode."""
        transformer = ClaudeAPIRequestTransformer()
        assert transformer.mode == "minimal"

    def test_init_invalid_mode_defaults_to_minimal(self) -> None:
        """Test invalid mode defaults to minimal."""
        transformer = ClaudeAPIRequestTransformer(mode="INVALID")
        assert transformer.mode == "minimal"


class TestHeaderTransformation:
    """Test header transformation functionality."""

    def test_transform_headers_basic_auth_injection(self) -> None:
        """Test basic authorization header injection and exclusion of problematic headers."""
        transformer = ClaudeAPIRequestTransformer()
        headers = {
            "content-type": "application/json",
            "host": "example.com",
            "x-api-key": "old-key",
            "authorization": "Bearer old-token",
        }

        result = transformer.transform_headers(headers, access_token="test-token")

        assert result["authorization"] == "Bearer test-token"
        assert result["content-type"] == "application/json"
        assert "host" not in result
        assert "x-api-key" not in result

    def test_transform_headers_no_access_token_raises_error(self) -> None:
        """Test that missing access_token raises RuntimeError."""
        transformer = ClaudeAPIRequestTransformer()
        headers = {"content-type": "application/json"}

        with pytest.raises(RuntimeError, match="access_token parameter is required"):
            transformer.transform_headers(headers)

    def test_transform_headers_with_detected_headers(self) -> None:
        """Test header transformation with detected headers from CLI."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field="test system"),
            raw_headers_ordered=[
                ("X-App", "cli"),
                ("anthropic-beta", "claude-code-20250219"),
            ],
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(detection_service=detection_service)
        headers = {"content-type": "application/json"}

        result = transformer.transform_headers(headers, access_token="test-token")

        assert result["X-App"] == "cli"
        assert result["anthropic-beta"] == "claude-code-20250219"
        assert result["authorization"] == "Bearer test-token"


class TestCacheControlDetection:
    """Test cache_control block detection functionality."""

    def test_find_cache_control_blocks_in_system_and_messages(self) -> None:
        """Test finding cache_control blocks in both system and messages."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "system": [
                {
                    "type": "text",
                    "text": "System",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "User 1",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {"type": "text", "text": "User 2"},
                        {
                            "type": "text",
                            "text": "User 3",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }

        blocks = transformer._find_cache_control_blocks(data)

        assert len(blocks) == 3
        assert ("system", 0, 0) in blocks
        assert ("message", 0, 0) in blocks
        assert ("message", 0, 2) in blocks

    def test_find_cache_control_blocks_in_tools(self) -> None:
        """Test finding cache_control blocks in tools field."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "tools": [
                {"name": "tool1", "cache_control": {"type": "ephemeral"}},
                {"name": "tool2"},
                {"name": "tool3", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

        blocks = transformer._find_cache_control_blocks(data)

        assert len(blocks) == 3
        assert ("tool", 0, 0) in blocks
        assert ("tool", 2, 0) in blocks
        assert ("message", 0, 0) in blocks


class TestCacheControlLimiting:
    """Test cache_control limiting functionality."""

    def test_limit_cache_control_blocks_removes_excess(self) -> None:
        """Test that excess cache_control blocks are removed from the end."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "system": [
                {
                    "type": "text",
                    "text": "System 1",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "User 1",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "User 2",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "User 3",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }

        result = transformer._limit_cache_control_blocks(data, max_blocks=2)

        # Should keep first 2 blocks and remove last 2
        assert "cache_control" in result["system"][0]
        assert "cache_control" in result["messages"][0]["content"][0]
        assert "cache_control" not in result["messages"][0]["content"][1]
        assert "cache_control" not in result["messages"][0]["content"][2]

    def test_limit_cache_control_blocks_preserves_original_data(self) -> None:
        """Test that original data is not modified."""
        transformer = ClaudeAPIRequestTransformer()
        original_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Message",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ]
        }

        transformer._limit_cache_control_blocks(original_data, max_blocks=0)

        # Original should be unchanged
        assert "cache_control" in original_data["messages"][0]["content"][0]

    def test_limit_cache_control_blocks_removes_from_tools(self) -> None:
        """Test that excess cache_control blocks are removed from tools."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "tools": [
                {"name": "tool1", "cache_control": {"type": "ephemeral"}},
                {"name": "tool2", "cache_control": {"type": "ephemeral"}},
                {"name": "tool3", "cache_control": {"type": "ephemeral"}},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            ],
        }

        result = transformer._limit_cache_control_blocks(data, max_blocks=2)

        # Should keep first 2 blocks and remove last 2
        assert "cache_control" in result["tools"][0]  # kept
        assert "cache_control" in result["tools"][1]  # kept
        assert "cache_control" not in result["tools"][2]  # removed
        assert "cache_control" not in result["messages"][0]["content"][0]  # removed


class TestSystemPromptInjection:
    """Test system prompt injection functionality."""

    def test_transform_body_mode_none_no_injection(self) -> None:
        """Test that 'none' mode doesn't inject system prompts."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field="Claude Code system prompt"),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="none"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        assert "system" not in result_data

    def test_transform_body_mode_minimal_string_system(self) -> None:
        """Test minimal mode with string system prompt."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(
                system_field="You are Claude Code.\nYou have many features."
            ),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="minimal"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should inject only first line
        assert result_data["system"] == "You are Claude Code."

    def test_transform_body_mode_minimal_list_system(self) -> None:
        """Test minimal mode with list system prompt preserves cache_control."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(
                system_field=[
                    {
                        "type": "text",
                        "text": "You are Claude Code",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": "Additional instructions"},
                ]
            ),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="minimal"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should inject only first element with cache_control preserved
        expected = [
            {
                "type": "text",
                "text": "You are Claude Code",
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert result_data["system"] == expected

    def test_transform_body_mode_full_complete_injection(self) -> None:
        """Test full mode injects complete system prompt."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        full_system = [
            {"type": "text", "text": "You are Claude Code"},
            {"type": "text", "text": "Follow instructions"},
        ]
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field=full_system),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="full"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        assert result_data["system"] == full_system

    def test_transform_body_prepends_to_existing_system(self) -> None:
        """Test system prompt prepending to existing system field."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field="You are Claude Code."),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="full"
        )
        body_data = {
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should convert both to list format with detected first
        expected = [
            {"type": "text", "text": "You are Claude Code."},
            {"type": "text", "text": "You are helpful."},
        ]
        assert result_data["system"] == expected

    def test_transform_body_invalid_json_returns_original(self) -> None:
        """Test body transformation with invalid JSON returns original."""
        transformer = ClaudeAPIRequestTransformer()
        invalid_body = b"{ invalid json }"

        result = transformer.transform_body(invalid_body)

        assert result == invalid_body

    def test_transform_body_integrates_cache_control_limiting(self) -> None:
        """Test body transformation includes cache_control limiting."""
        transformer = ClaudeAPIRequestTransformer()
        body_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Msg 1",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "Msg 2",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "Msg 3",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "Msg 4",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "Msg 5",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ]
        }
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Should limit to 4 cache_control blocks (last one removed)
        content = result_data["messages"][0]["content"]
        assert "cache_control" in content[0]  # kept
        assert "cache_control" in content[1]  # kept
        assert "cache_control" in content[2]  # kept
        assert "cache_control" in content[3]  # kept
        assert "cache_control" not in content[4]  # removed
