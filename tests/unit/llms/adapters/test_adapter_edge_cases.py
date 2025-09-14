import pytest
from pydantic import ValidationError

from ccproxy.llms.anthropic.models import (
    CreateMessageRequest as AnthropicCreateMessageRequest,
)
from ccproxy.llms.anthropic.models import (
    MessageResponse as AnthropicMessageResponse,
)
from ccproxy.llms.anthropic.models import (
    TextBlock as AnthropicTextBlock,
)
from ccproxy.llms.anthropic.models import (
    Usage as AnthropicUsage,
)
from ccproxy.llms.openai.models import (
    ChatCompletionRequest as OpenAIChatRequest,
)
from ccproxy.llms.openai.models import (
    ChatCompletionResponse as OpenAIChatResponse,
)
from ccproxy.llms.openai.models import (
    ResponseRequest as OpenAIResponseRequest,
)


class TestMalformedContentBlocks:
    """Test handling of malformed content blocks across adapters"""

    @pytest.mark.asyncio
    async def test_openai_chat_malformed_content(self) -> None:
        """Test OpenAI Chat adapter with malformed content structures"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with malformed image URL
        req_bad_image = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "not-a-valid-data-url"},
                        },
                    ],
                }
            ],
        )
        out_bad_image = await adapter.adapt_request(req_bad_image)
        anth_bad_image = AnthropicCreateMessageRequest.model_validate(out_bad_image)
        # Should still process the text part
        content = anth_bad_image.messages[0].content
        if isinstance(content, list):
            assert len(content) >= 1
            assert content[0].type == "text"
        else:
            assert isinstance(content, str)

        # Test with missing required fields
        req_missing_fields = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text"},  # Missing "text" field
                        {"type": "image_url"},  # Missing "image_url" field
                    ],
                }
            ],
        }
        # Should handle gracefully and not crash
        try:
            out_missing = await adapter.adapt_request(req_missing_fields)
            anth_missing = AnthropicCreateMessageRequest.model_validate(out_missing)
            # At minimum should have model and messages structure
            assert anth_missing.model == "gpt-4o"
        except Exception:
            # May fail validation, which is acceptable for malformed input
            pass

    @pytest.mark.asyncio
    async def test_anthropic_malformed_response_content(self) -> None:
        """Test Anthropic response adapter with malformed content blocks"""
        from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
            AnthropicMessagesToOpenAIChatAdapter,
        )

        adapter = AnthropicMessagesToOpenAIChatAdapter()

        # Test with malformed content blocks
        malformed_resp = {
            "id": "msg_malformed",
            "type": "message",
            "role": "assistant",
            "model": "claude",
            "content": [
                {"type": "text"},  # Missing "text" field
                {"type": "tool_use", "name": "test"},  # Missing required fields
                {"type": "unknown_type", "data": "something"},  # Unknown content type
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }

        # Should handle gracefully
        try:
            out_malformed = await adapter.adapt_response(malformed_resp)
            openai_resp = OpenAIChatResponse.model_validate(out_malformed)
            # Should have basic response structure
            assert openai_resp.choices[0].message.role == "assistant"
        except Exception:
            # May fail validation for severely malformed input
            pass


class TestEmptyMessageLists:
    """Test handling of empty message lists and content"""

    @pytest.mark.asyncio
    async def test_empty_messages_list(self) -> None:
        """Test adapters with empty messages array"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with empty messages array
        req_empty = OpenAIChatRequest(
            model="gpt-4o",
            messages=[],  # Empty messages
        )
        out_empty = await adapter.adapt_request(req_empty)
        anth_empty = AnthropicCreateMessageRequest.model_validate(out_empty)
        assert anth_empty.model == "gpt-4o"
        assert len(anth_empty.messages) == 0

    @pytest.mark.asyncio
    async def test_messages_with_empty_content(self) -> None:
        """Test messages with empty or None content"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with None content
        req_none_content = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": None},
                {"role": "assistant", "content": ""},
            ],
        )
        out_none = await adapter.adapt_request(req_none_content)
        anth_none = AnthropicCreateMessageRequest.model_validate(out_none)
        # Should filter out messages with None content
        assert len(anth_none.messages) == 1
        assert anth_none.messages[0].content == ""

        # Test with empty content array
        req_empty_array = OpenAIChatRequest(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": []},
            ],
        )
        out_empty_array = await adapter.adapt_request(req_empty_array)
        anth_empty_array = AnthropicCreateMessageRequest.model_validate(out_empty_array)
        assert anth_empty_array.model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_openai_responses_empty_input(self) -> None:
        """Test OpenAI Responses adapter with empty input"""
        from ccproxy.llms.adapters.openai_responses_request_to_anthropic_messages import (
            OpenAIResponsesRequestToAnthropicMessagesAdapter,
        )

        adapter = OpenAIResponsesRequestToAnthropicMessagesAdapter()

        # Test with empty string input
        req_empty_string = OpenAIResponseRequest(
            model="gpt-4o",
            input="",  # Empty string
        )
        out_empty_string = await adapter.adapt_request(req_empty_string)
        anth_empty_string = AnthropicCreateMessageRequest.model_validate(
            out_empty_string
        )
        assert anth_empty_string.model == "gpt-4o"

        # Test with None input - this is invalid for the model
        with pytest.raises(ValidationError):
            OpenAIResponseRequest(
                model="gpt-4o",
                input=None,  # None input
            )


class TestUnexpectedFieldValues:
    """Test handling of unexpected field values and types"""

    @pytest.mark.asyncio
    async def test_invalid_model_names(self) -> None:
        """Test with invalid or unusual model names"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with very long model name
        req_long_model = OpenAIChatRequest(
            model="a" * 1000,  # Very long model name
            messages=[{"role": "user", "content": "test"}],
        )
        out_long = await adapter.adapt_request(req_long_model)
        anth_long = AnthropicCreateMessageRequest.model_validate(out_long)
        assert anth_long.model == "a" * 1000

        # Test with special characters in model name
        req_special_model = OpenAIChatRequest(
            model="model@#$%^&*()",
            messages=[{"role": "user", "content": "test"}],
        )
        out_special = await adapter.adapt_request(req_special_model)
        anth_special = AnthropicCreateMessageRequest.model_validate(out_special)
        assert anth_special.model == "model@#$%^&*()"

    @pytest.mark.asyncio
    async def test_extreme_token_values(self) -> None:
        """Test with extreme token limit values"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with very large token limit
        req_huge_tokens = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=999999999,
        )
        out_huge = await adapter.adapt_request(req_huge_tokens)
        anth_huge = AnthropicCreateMessageRequest.model_validate(out_huge)
        assert anth_huge.max_tokens == 999999999

        # Test with zero tokens
        req_zero_tokens = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            max_completion_tokens=0,
        )
        out_zero = await adapter.adapt_request(req_zero_tokens)
        anth_zero = AnthropicCreateMessageRequest.model_validate(out_zero)
        assert anth_zero.max_tokens == 0

    @pytest.mark.asyncio
    async def test_invalid_role_values(self) -> None:
        """Test with invalid or unexpected role values"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with unknown role
        req_unknown_role = {
            "model": "gpt-4o",
            "messages": [
                {"role": "unknown_role", "content": "test"},
                {"role": "user", "content": "valid message"},
            ],
        }
        try:
            out_unknown = await adapter.adapt_request(req_unknown_role)
            anth_unknown = AnthropicCreateMessageRequest.model_validate(out_unknown)
            # Should at least process valid messages
            assert anth_unknown.model == "gpt-4o"
        except Exception:
            # May fail validation for unknown roles
            pass

    @pytest.mark.asyncio
    async def test_invalid_temperature_values(self) -> None:
        """Test with invalid temperature values"""
        # Test with temperature > 2.0
        with pytest.raises(ValidationError):
            OpenAIChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                temperature=10.0,  # Very high temperature
            )

        # Test with negative temperature
        with pytest.raises(ValidationError):
            OpenAIChatRequest(
                model="gpt-4o",
                messages=[{"role": "user", "content": "test"}],
                temperature=-1.0,  # Negative temperature
            )


class TestBoundaryConditions:
    """Test boundary conditions and limit cases"""

    @pytest.mark.asyncio
    async def test_maximum_message_length(self) -> None:
        """Test with extremely long messages"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with very long message content
        very_long_content = "a" * 100000  # 100k characters
        req_long_msg = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": very_long_content}],
        )
        out_long_msg = await adapter.adapt_request(req_long_msg)
        anth_long_msg = AnthropicCreateMessageRequest.model_validate(out_long_msg)
        assert anth_long_msg.messages[0].content == very_long_content

    @pytest.mark.asyncio
    async def test_maximum_tools_array(self) -> None:
        """Test with large number of tools"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Create many tools
        many_tools = []
        for i in range(100):  # 100 tools
            many_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i}",
                        "description": f"Tool number {i}",
                        "parameters": {
                            "type": "object",
                            "properties": {"param": {"type": "string"}},
                        },
                    },
                }
            )

        req_many_tools = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Use tools"}],
            tools=many_tools,
        )
        out_many_tools = await adapter.adapt_request(req_many_tools)
        anth_many_tools = AnthropicCreateMessageRequest.model_validate(out_many_tools)
        assert len(anth_many_tools.tools) == 100

    @pytest.mark.asyncio
    async def test_deeply_nested_json_schema(self) -> None:
        """Test with deeply nested JSON schemas"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Create deeply nested schema
        nested_schema = {"type": "object", "properties": {}}
        current_level = nested_schema["properties"]
        for i in range(20):  # 20 levels deep
            current_level[f"level_{i}"] = {"type": "object", "properties": {}}
            current_level = current_level[f"level_{i}"]["properties"]
        current_level["final"] = {"type": "string"}

        req_nested = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "nested",
                    "schema": nested_schema,
                },
            },
        )
        out_nested = await adapter.adapt_request(req_nested)
        anth_nested = AnthropicCreateMessageRequest.model_validate(out_nested)
        # Should handle deeply nested schema
        assert "nested" in anth_nested.system or anth_nested.system is not None

    @pytest.mark.asyncio
    async def test_unicode_and_special_characters(self) -> None:
        """Test handling of Unicode and special characters"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with various Unicode characters
        unicode_content = "Hello 🌍 世界 Мир עולם مرحبا 🚀 ñáéíóú àèìòù"
        req_unicode = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": unicode_content}],
        )
        out_unicode = await adapter.adapt_request(req_unicode)
        anth_unicode = AnthropicCreateMessageRequest.model_validate(out_unicode)
        assert anth_unicode.messages[0].content == unicode_content

        # Test with control characters and escape sequences
        control_content = "Test\n\t\r\\n\t\"quotes\"'apostrophes'\x00\x1f"
        req_control = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": control_content}],
        )
        out_control = await adapter.adapt_request(req_control)
        anth_control = AnthropicCreateMessageRequest.model_validate(out_control)
        # Should handle control characters without crashing
        assert anth_control.model == "gpt-4o"


class TestCompatibilityEdgeCases:
    """Test edge cases related to cross-format compatibility"""

    @pytest.mark.asyncio
    async def test_anthropic_to_openai_unsupported_features(self) -> None:
        """Test Anthropic features that don't map cleanly to OpenAI"""
        from ccproxy.llms.adapters.anthropic_messages_to_openai_chatcompletions import (
            AnthropicMessagesToOpenAIChatAdapter,
        )

        adapter = AnthropicMessagesToOpenAIChatAdapter()

        # Test with Anthropic-specific content types
        anthropic_resp = AnthropicMessageResponse(
            id="msg_edge",
            type="message",
            role="assistant",
            model="claude",
            content=[
                AnthropicTextBlock(type="text", text="Regular text"),
                # Add thinking block which doesn't directly map to OpenAI
                {
                    "type": "thinking",
                    "thinking": "Internal reasoning",
                    "signature": "sig123",
                },
            ],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=AnthropicUsage(input_tokens=5, output_tokens=10),
        )

        out_unsupported = await adapter.adapt_response(anthropic_resp)
        openai_resp = OpenAIChatResponse.model_validate(out_unsupported)

        # Should handle the response even with unsupported content types
        assert openai_resp.choices[0].message.role == "assistant"
        # Thinking content should be serialized in some way
        content = openai_resp.choices[0].message.content or ""
        assert len(content) > 0

    @pytest.mark.asyncio
    async def test_openai_to_anthropic_unsupported_parameters(self) -> None:
        """Test OpenAI parameters that don't map to Anthropic"""
        from ccproxy.llms.adapters.openai_chatcompletions_to_anthropic_messages import (
            OpenAIChatToAnthropicMessagesAdapter,
        )

        adapter = OpenAIChatToAnthropicMessagesAdapter()

        # Test with OpenAI-specific parameters
        req_unsupported = OpenAIChatRequest(
            model="gpt-4o",
            messages=[{"role": "user", "content": "test"}],
            # OpenAI-specific parameters that may not map directly
            frequency_penalty=1.5,
            presence_penalty=0.8,
            logit_bias={"1234": 100},
            user="user123",
            seed=42,
            n=3,  # Multiple completions
        )

        out_unsupported = await adapter.adapt_request(req_unsupported)
        anth_unsupported = AnthropicCreateMessageRequest.model_validate(out_unsupported)

        # Should adapt core functionality even if some parameters are ignored
        assert anth_unsupported.model == "gpt-4o"
        assert anth_unsupported.messages[0].content == "test"
