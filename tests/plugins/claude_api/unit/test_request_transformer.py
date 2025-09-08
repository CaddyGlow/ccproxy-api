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

    def test_find_cache_control_blocks_in_tool_use_and_tool_result(self) -> None:
        """Test finding cache_control blocks in tool_use and tool_result content blocks."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "tool_use",
                            "name": "search",
                            "input": {"query": "test"},
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_result",
                            "content": "search results",
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
            ],
        }

        blocks = transformer._find_cache_control_blocks(data)

        assert len(blocks) == 3
        assert ("message", 0, 0) in blocks  # text block
        assert ("tool_use", 0, 1) in blocks  # tool_use block
        assert ("tool_result", 1, 0) in blocks  # tool_result block


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
        """Test that excess cache_control blocks are removed from tools using smart algorithm."""
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

        # Smart algorithm should keep the 2 largest blocks by content size
        # All tools have identical size ("tool1", "tool2", "tool3" are all 5 chars)
        # So it should keep first 2 tools by discovery order
        total_cache_blocks = sum(
            [
                1 if "cache_control" in result["tools"][0] else 0,
                1 if "cache_control" in result["tools"][1] else 0,
                1 if "cache_control" in result["tools"][2] else 0,
                1 if "cache_control" in result["messages"][0]["content"][0] else 0,
            ]
        )
        assert total_cache_blocks == 2  # Only 2 blocks should have cache_control


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

        # Should inject only first line (metadata cleaned from final output)
        expected = [{"type": "text", "text": "You are Claude Code."}]
        assert result_data["system"] == expected

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

        # Should inject only first element with cache_control preserved (metadata cleaned from final output)
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

        # Should inject complete system (metadata cleaned from final output)
        expected = [
            {"type": "text", "text": "You are Claude Code"},
            {"type": "text", "text": "Follow instructions"},
        ]
        assert result_data["system"] == expected

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

        # Should convert both to list format with detected first (metadata cleaned from final output)
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


class TestSystemPromptMarking:
    """Test system prompt marking functionality."""

    def test_mark_injected_system_prompts_string_input(self) -> None:
        """Test marking string system prompt."""
        transformer = ClaudeAPIRequestTransformer()

        result = transformer._mark_injected_system_prompts("You are Claude Code")

        expected = [
            {"type": "text", "text": "You are Claude Code", "_ccproxy_injected": True}
        ]
        assert result == expected

    def test_mark_injected_system_prompts_list_input(self) -> None:
        """Test marking list system prompt."""
        transformer = ClaudeAPIRequestTransformer()

        input_data = [
            {
                "type": "text",
                "text": "System prompt 1",
                "cache_control": {"type": "ephemeral"},
            },
            {"type": "text", "text": "System prompt 2"},
        ]

        result = transformer._mark_injected_system_prompts(input_data)

        expected = [
            {
                "type": "text",
                "text": "System prompt 1",
                "cache_control": {"type": "ephemeral"},
                "_ccproxy_injected": True,
            },
            {"type": "text", "text": "System prompt 2", "_ccproxy_injected": True},
        ]
        assert result == expected

    def test_transform_body_cleans_injected_metadata_from_final_output(self) -> None:
        """Test that transform_body removes _ccproxy_injected metadata from final output."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field="You are Claude Code"),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="full"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Final output should NOT contain internal metadata (cleaned for API)
        assert "_ccproxy_injected" not in result_data["system"][0]
        assert result_data["system"][0]["text"] == "You are Claude Code"


class TestSmartCacheControlLimiting:
    """Test smart cache control limiting algorithm."""

    def test_smart_limiting_preserves_injected_system_prompts(self) -> None:
        """Test that injected system prompts are always preserved."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "system": [
                {
                    "type": "text",
                    "text": "Injected system prompt",
                    "cache_control": {"type": "ephemeral"},
                    "_ccproxy_injected": True,
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "User message 1",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "User message 2",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "User message 3",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "User message 4",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }

        result = transformer._limit_cache_control_blocks(data, max_blocks=3)

        # Injected system prompt should always be preserved
        assert "cache_control" in result["system"][0]
        assert result["system"][0]["_ccproxy_injected"] is True

        # Should keep 2 largest user messages (by content size)
        content = result["messages"][0]["content"]
        cache_blocks_kept = sum(1 for block in content if "cache_control" in block)
        assert cache_blocks_kept == 2  # Only 2 non-injected blocks kept

    def test_smart_limiting_keeps_largest_blocks_by_size(self) -> None:
        """Test that smart limiting keeps the largest blocks by content size."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Small",  # 5 chars
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "This is a much longer message with more content",  # 47 chars
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "Medium length message",  # 21 chars
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "tool_use",
                            "name": "search",
                            "input": {
                                "query": "This is a very detailed search query with lots of parameters"
                            },  # Large input
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }

        result = transformer._limit_cache_control_blocks(data, max_blocks=2)

        # Should keep the 2 largest blocks: tool_use (largest) and long text message
        content = result["messages"][0]["content"]
        assert "cache_control" not in content[0]  # "Small" - removed
        assert "cache_control" in content[1]  # Long text - kept
        assert "cache_control" not in content[2]  # Medium - removed
        assert "cache_control" in content[3]  # tool_use - kept (largest)

    def test_smart_limiting_handles_mixed_injected_and_regular_blocks(self) -> None:
        """Test smart limiting with mix of injected and regular blocks."""
        transformer = ClaudeAPIRequestTransformer()
        data = {
            "system": [
                {
                    "type": "text",
                    "text": "Injected system 1",
                    "cache_control": {"type": "ephemeral"},
                    "_ccproxy_injected": True,
                },
                {
                    "type": "text",
                    "text": "Injected system 2",
                    "cache_control": {"type": "ephemeral"},
                    "_ccproxy_injected": True,
                },
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Short message",
                            "cache_control": {"type": "ephemeral"},
                        },
                        {
                            "type": "text",
                            "text": "This is a very long user message with extensive content that should be prioritized",
                            "cache_control": {"type": "ephemeral"},
                        },
                    ],
                }
            ],
        }

        result = transformer._limit_cache_control_blocks(data, max_blocks=3)

        # Should preserve both injected system prompts + 1 largest user message
        assert "cache_control" in result["system"][0]  # injected - preserved
        assert "cache_control" in result["system"][1]  # injected - preserved

        content = result["messages"][0]["content"]
        assert "cache_control" not in content[0]  # short message - removed
        assert "cache_control" in content[1]  # long message - kept

    def test_calculate_content_size_various_block_types(self) -> None:
        """Test content size calculation for different block types."""
        transformer = ClaudeAPIRequestTransformer()

        # Test text block
        text_block = {"type": "text", "text": "Hello world"}
        assert transformer._calculate_content_size(text_block) == 11

        # Test tool_use block
        tool_use_block = {
            "type": "tool_use",
            "name": "search",
            "input": {"query": "test"},
        }
        # "search" (6) + str({"query": "test"}) (17) = 23
        assert transformer._calculate_content_size(tool_use_block) == 23

        # Test tool_result block with string content
        tool_result_block = {"type": "tool_result", "content": "Search results"}
        assert transformer._calculate_content_size(tool_result_block) == 14

    def test_clean_internal_metadata_removes_ccproxy_injected(self) -> None:
        """Test that internal metadata is cleaned from request data."""
        transformer = ClaudeAPIRequestTransformer()

        data = {
            "system": [
                {
                    "type": "text",
                    "text": "System prompt",
                    "cache_control": {"type": "ephemeral"},
                    "_ccproxy_injected": True,
                }
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "User message",
                            "_ccproxy_injected": True,  # This shouldn't normally happen but test it
                        }
                    ],
                }
            ],
            "tools": [
                {
                    "name": "test_tool",
                    "_ccproxy_injected": True,  # This shouldn't normally happen but test it
                }
            ],
        }

        result = transformer._clean_internal_metadata(data)

        # Original data should be unchanged
        assert data["system"][0]["_ccproxy_injected"] is True

        # Cleaned data should not have _ccproxy_injected
        assert "_ccproxy_injected" not in result["system"][0]
        assert "_ccproxy_injected" not in result["messages"][0]["content"][0]
        assert "_ccproxy_injected" not in result["tools"][0]

        # Other fields should be preserved
        assert result["system"][0]["text"] == "System prompt"
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

    def test_transform_body_cleans_metadata_in_final_output(self) -> None:
        """Test that transform_body removes internal metadata from final output."""
        detection_service = Mock(spec=ClaudeAPIDetectionService)
        cache_data = ClaudeCacheData(
            claude_version="1.0.60",
            headers=ClaudeCodeHeaders(),
            system_prompt=SystemPromptData(system_field="You are Claude Code"),
        )
        detection_service.get_cached_data.return_value = cache_data

        transformer = ClaudeAPIRequestTransformer(
            detection_service=detection_service, mode="full"
        )
        body_data = {"messages": [{"role": "user", "content": "Hello"}]}
        body = json.dumps(body_data).encode("utf-8")

        result = transformer.transform_body(body)
        result_data = json.loads(result.decode("utf-8"))

        # Final output should not contain internal metadata
        assert "_ccproxy_injected" not in result_data["system"][0]

        # But should contain the injected system prompt text
        assert result_data["system"][0]["text"] == "You are Claude Code"
