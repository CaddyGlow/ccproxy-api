"""Tests for Claude client."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_code_sdk import ClaudeCodeOptions

from claude_code_proxy.exceptions import ClaudeProxyError, ServiceUnavailableError
from claude_code_proxy.services.claude_client import ClaudeClient


class TestClaudeClient:
    """Test ClaudeClient class."""

    def test_init(self):
        """Test client initialization."""
        client = ClaudeClient()
        assert isinstance(client, ClaudeClient)

    def test_format_messages_to_prompt(self):
        """Test _format_messages_to_prompt method."""
        client = ClaudeClient()

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        prompt = client._format_messages_to_prompt(messages)

        expected = "Human: Hello\n\nAssistant: Hi there!\n\nHuman: How are you?"
        assert prompt == expected

    def test_format_messages_with_content_blocks(self):
        """Test _format_messages_to_prompt with content blocks."""
        client = ClaudeClient()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "text", "text": "Please describe it."},
                ],
            }
        ]

        prompt = client._format_messages_to_prompt(messages)

        expected = "Human: What's in this image? Please describe it."
        assert prompt == expected

    def test_format_messages_with_system(self):
        """Test _format_messages_to_prompt skips system messages."""
        client = ClaudeClient()

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ]

        prompt = client._format_messages_to_prompt(messages)

        expected = "Human: Hello"
        assert prompt == expected

    def test_extract_text_from_content(self):
        """Test _extract_text_from_content method."""
        from claude_code_sdk import TextBlock, ToolResultBlock, ToolUseBlock

        client = ClaudeClient()

        # Mock content blocks
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Hello world"

        tool_use_block = MagicMock(spec=ToolUseBlock)
        tool_use_block.name = "calculator"

        tool_result_block = MagicMock(spec=ToolResultBlock)
        tool_result_block.content = "Result: 42"

        content = [text_block, tool_use_block, tool_result_block]

        result = client._extract_text_from_content(content)  # type: ignore[arg-type]

        expected = "Hello world [Tool: calculator] Result: 42"
        assert result == expected

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_create_completion_non_streaming(self, mock_query):
        """Test create_completion for non-streaming."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        # Mock messages
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Hello there!"

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [text_block]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.session_id = "test_session_123"

        async def mock_async_iter():
            yield assistant_msg
            yield result_msg

        mock_query.return_value = mock_async_iter()

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await client.create_completion(messages, options, stream=False)

        assert isinstance(result, dict)
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["id"] == "msg_test_session_123"
        assert result["content"][0]["text"] == "Hello there!"

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_create_completion_streaming(self, mock_query):
        """Test create_completion for streaming."""
        from claude_code_sdk import AssistantMessage, ResultMessage, TextBlock

        # Mock streaming messages
        text_block1 = MagicMock(spec=TextBlock)
        text_block1.text = "Hello"

        text_block2 = MagicMock(spec=TextBlock)
        text_block2.text = " world!"

        assistant_msg1 = MagicMock(spec=AssistantMessage)
        assistant_msg1.content = [text_block1]

        assistant_msg2 = MagicMock(spec=AssistantMessage)
        assistant_msg2.content = [text_block2]

        result_msg = MagicMock(spec=ResultMessage)
        result_msg.session_id = "test_session_123"

        async def mock_async_query(*args, **kwargs):
            for msg in [assistant_msg1, assistant_msg2, result_msg]:
                yield msg

        mock_query.return_value = mock_async_query()

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        messages = [{"role": "user", "content": "Hello"}]

        result = await client.create_completion(messages, options, stream=True)

        # Should return an async iterator
        chunks = []
        async for chunk in result:  # type: ignore
            chunks.append(chunk)

        assert len(chunks) >= 2  # Should have multiple chunks

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Test list_models method."""
        client = ClaudeClient()

        models = await client.list_models()

        assert isinstance(models, list)
        # Current implementation returns empty list
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with ClaudeClient() as client:
            assert isinstance(client, ClaudeClient)

        # Should not raise any errors

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_cli_not_found_error(self, mock_query):
        """Test handling of CLI not found error."""
        from claude_code_sdk import CLINotFoundError

        mock_query.side_effect = CLINotFoundError("Claude CLI not found")

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        with pytest.raises(ServiceUnavailableError) as exc_info:
            await client.create_completion(
                [{"role": "user", "content": "test"}], options
            )

        assert "Claude CLI not available" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_process_error(self, mock_query):
        """Test handling of process error."""
        from claude_code_sdk import ProcessError

        mock_query.side_effect = ProcessError("Process failed")

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        with pytest.raises(ClaudeProxyError) as exc_info:
            await client.create_completion(
                [{"role": "user", "content": "test"}], options
            )

        assert "Claude process error" in str(exc_info.value)
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_unexpected_error(self, mock_query):
        """Test handling of unexpected error."""
        mock_query.side_effect = RuntimeError("Unexpected error")

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        with pytest.raises(ClaudeProxyError) as exc_info:
            await client.create_completion(
                [{"role": "user", "content": "test"}], options
            )

        assert "Unexpected error" in str(exc_info.value)
        assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_no_result_message_error(self, mock_query):
        """Test handling when no result message is received."""
        from claude_code_sdk import AssistantMessage, TextBlock

        from claude_code_proxy.services.claude_client import ClaudeClientError

        # Mock only assistant message, no result message
        text_block = MagicMock(spec=TextBlock)
        text_block.text = "Hello there!"

        assistant_msg = MagicMock(spec=AssistantMessage)
        assistant_msg.content = [text_block]

        async def mock_async_iter():
            yield assistant_msg

        mock_query.return_value = mock_async_iter()

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        with pytest.raises(ClaudeClientError) as exc_info:
            await client.create_completion(
                [{"role": "user", "content": "test"}], options
            )

        assert "No result message received" in str(exc_info.value)

    @pytest.mark.asyncio
    @patch("claude_code_proxy.services.claude_client.query")
    async def test_no_assistant_message_error(self, mock_query):
        """Test handling when no assistant message is received."""
        from claude_code_sdk import ResultMessage

        from claude_code_proxy.services.claude_client import ClaudeClientError

        # Mock only result message, no assistant message
        result_msg = MagicMock(spec=ResultMessage)
        result_msg.session_id = "test_session_123"

        async def mock_async_iter():
            yield result_msg

        mock_query.return_value = mock_async_iter()

        client = ClaudeClient()
        options = ClaudeCodeOptions(
            model="claude-3-5-sonnet-20241022", permission_mode="default"
        )

        with pytest.raises(ClaudeClientError) as exc_info:
            await client.create_completion(
                [{"role": "user", "content": "test"}], options
            )

        assert "No assistant response received" in str(exc_info.value)
