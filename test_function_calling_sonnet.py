"""Comprehensive tests for function calling support in ResponseAdapter.

This test suite covers all aspects of function calling transformation between
Anthropic Messages API and OpenAI Response API formats.
"""

import pytest

from ccproxy.adapters.openai.response_adapter import ResponseAdapter
from ccproxy.adapters.openai.response_models import (
    ResponseTool,
    ResponseToolChoice,
)


class TestFunctionCallingRequestTransformation:
    """Test conversion from Chat Completions to Response API with function calling."""

    def setup_method(self):
        self.adapter = ResponseAdapter()

    def test_convert_simple_tools_to_response_api(self):
        """Test basic tools conversion."""
        chat_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get current weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                            "required": ["location"],
                        },
                    },
                }
            ],
        }

        response_request = self.adapter.chat_to_response_request(chat_request)

        assert response_request.tools is not None
        assert len(response_request.tools) == 1
        tool = response_request.tools[0]
        assert tool.type == "function"
        assert tool.function.name == "get_weather"
        assert tool.function.description == "Get current weather"
        assert tool.function.parameters["type"] == "object"
        assert "location" in tool.function.parameters["properties"]

    def test_convert_multiple_tools_to_response_api(self):
        """Test conversion of multiple tools."""
        chat_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Help me with tasks"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "description": "Send an email",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string"},
                                "subject": {"type": "string"},
                                "body": {"type": "string"},
                            },
                            "required": ["to", "subject"],
                        },
                    },
                },
            ],
        }

        response_request = self.adapter.chat_to_response_request(chat_request)

        assert response_request.tools is not None
        assert len(response_request.tools) == 2

        weather_tool = response_request.tools[0]
        assert weather_tool.function.name == "get_weather"

        email_tool = response_request.tools[1]
        assert email_tool.function.name == "send_email"
        assert len(email_tool.function.parameters["required"]) == 2

    def test_convert_tool_choice_variants(self):
        """Test all tool_choice conversion variants."""
        base_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "test_func", "parameters": {}},
                }
            ],
        }

        # Test auto
        request = {**base_request, "tool_choice": "auto"}
        response = self.adapter.chat_to_response_request(request)
        assert response.tool_choice == "auto"

        # Test none
        request = {**base_request, "tool_choice": "none"}
        response = self.adapter.chat_to_response_request(request)
        assert response.tool_choice == "none"

        # Test required (maps to "any" in Anthropic)
        request = {**base_request, "tool_choice": "required"}
        response = self.adapter.chat_to_response_request(request)
        assert response.tool_choice == "required"

        # Test specific function
        request = {
            **base_request,
            "tool_choice": {"type": "function", "function": {"name": "test_func"}},
        }
        response = self.adapter.chat_to_response_request(request)
        assert isinstance(response.tool_choice, ResponseToolChoice)
        assert response.tool_choice.type == "function"
        assert response.tool_choice.function["name"] == "test_func"

    def test_convert_messages_with_tool_calls(self):
        """Test conversion of assistant messages with tool calls."""
        chat_request = {
            "model": "gpt-4",
            "messages": [
                {"role": "user", "content": "Get the weather"},
                {
                    "role": "assistant",
                    "content": "I'll check the weather for you.",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco"}',
                            },
                        }
                    ],
                },
            ],
        }

        response_request = self.adapter.chat_to_response_request(chat_request)

        # Find the assistant message
        assistant_msg = None
        for msg in response_request.input:
            if msg.role == "assistant":
                assistant_msg = msg
                break

        assert assistant_msg is not None
        assert len(assistant_msg.content) == 2  # Text + tool call

        # Check text content
        text_content = [c for c in assistant_msg.content if c.type == "output_text"]
        assert len(text_content) == 1
        assert text_content[0].text == "I'll check the weather for you."

        # Check tool call content
        tool_content = [c for c in assistant_msg.content if c.type == "tool_call"]
        assert len(tool_content) == 1
        assert tool_content[0].id == "call_123"
        assert tool_content[0].function.name == "get_weather"
        assert tool_content[0].function.arguments == '{"location": "San Francisco"}'

    def test_preserve_other_parameters(self):
        """Test that other parameters are preserved correctly."""
        chat_request = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "Test"}],
            "tools": [
                {"type": "function", "function": {"name": "test", "parameters": {}}}
            ],
            "parallel_tool_calls": True,
            "max_tool_calls": 5,
        }

        response_request = self.adapter.chat_to_response_request(chat_request)

        assert response_request.parallel_tool_calls is True
        assert response_request.max_tool_calls == 5
        assert response_request.stream is True  # Always true for Response API
        assert response_request.store is False  # Always false for Response API


class TestFunctionCallingResponseTransformation:
    """Test conversion from Response API to Chat Completions with function calls."""

    def setup_method(self):
        self.adapter = ResponseAdapter()

    def test_convert_response_with_tool_calls(self):
        """Test conversion of Response API response with tool calls."""
        response_data = {
            "id": "resp_123",
            "object": "response",
            "created_at": 1234567890,
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I'll check the weather for you.",
                        },
                        {
                            "type": "tool_call",
                            "id": "call_abc123",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                            },
                        },
                    ],
                }
            ],
            "usage": {"input_tokens": 50, "output_tokens": 25, "total_tokens": 75},
        }

        chat_response = self.adapter.response_to_chat_completion(response_data)

        # Check basic response structure
        assert chat_response.id == "resp_123"
        assert chat_response.model == "gpt-5"
        assert len(chat_response.choices) == 1

        choice = chat_response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert choice.message.role == "assistant"
        assert choice.message.content == "I'll check the weather for you."

        # Check tool calls
        assert choice.message.tool_calls is not None
        assert len(choice.message.tool_calls) == 1

        tool_call = choice.message.tool_calls[0]
        assert tool_call.id == "call_abc123"
        assert tool_call.type == "function"
        assert tool_call.function.name == "get_weather"
        assert (
            tool_call.function.arguments
            == '{"location": "San Francisco", "unit": "celsius"}'
        )

        # Check usage
        assert chat_response.usage.prompt_tokens == 50
        assert chat_response.usage.completion_tokens == 25
        assert chat_response.usage.total_tokens == 75

    def test_convert_response_multiple_tool_calls(self):
        """Test conversion with multiple tool calls."""
        response_data = {
            "id": "resp_456",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "I'll help you with both tasks.",
                        },
                        {
                            "type": "tool_call",
                            "id": "call_1",
                            "function": {"name": "get_weather", "arguments": "{}"},
                        },
                        {
                            "type": "tool_call",
                            "id": "call_2",
                            "function": {
                                "name": "send_email",
                                "arguments": '{"to": "test@example.com"}',
                            },
                        },
                    ],
                }
            ],
        }

        chat_response = self.adapter.response_to_chat_completion(response_data)

        choice = chat_response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 2

        # Check both tool calls
        tool_calls = choice.message.tool_calls
        assert tool_calls[0].id == "call_1"
        assert tool_calls[0].function.name == "get_weather"
        assert tool_calls[1].id == "call_2"
        assert tool_calls[1].function.name == "send_email"

    def test_convert_response_without_tool_calls(self):
        """Test conversion of regular response without tool calls."""
        response_data = {
            "id": "resp_789",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": "Here's your answer."},
                    ],
                }
            ],
        }

        chat_response = self.adapter.response_to_chat_completion(response_data)

        choice = chat_response.choices[0]
        assert choice.finish_reason == "stop"  # No tool calls
        assert choice.message.content == "Here's your answer."
        assert choice.message.tool_calls is None

    def test_convert_wrapped_response_format(self):
        """Test conversion of wrapped response format."""
        wrapped_response = {
            "type": "response.completed",
            "response": {
                "id": "resp_wrapped",
                "model": "gpt-5",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "Wrapped response"},
                            {
                                "type": "tool_call",
                                "id": "call_wrapped",
                                "function": {"name": "test_func", "arguments": "{}"},
                            },
                        ],
                    }
                ],
            },
        }

        chat_response = self.adapter.response_to_chat_completion(wrapped_response)

        assert chat_response.id == "resp_wrapped"
        choice = chat_response.choices[0]
        assert choice.finish_reason == "tool_calls"
        assert len(choice.message.tool_calls) == 1

    def test_malformed_json_arguments_handling(self):
        """Test handling of malformed JSON in function arguments."""
        response_data = {
            "id": "resp_malformed",
            "model": "gpt-5",
            "output": [
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "tool_call",
                            "id": "call_malformed",
                            "function": {
                                "name": "test_func",
                                "arguments": '{"invalid": json}',  # Invalid JSON
                            },
                        },
                    ],
                }
            ],
        }

        # Should not raise an exception, should preserve the string
        chat_response = self.adapter.response_to_chat_completion(response_data)

        choice = chat_response.choices[0]
        tool_call = choice.message.tool_calls[0]
        assert (
            tool_call.function.arguments == "{}"
        )  # Should be empty dict due to JSON parse failure


class TestFunctionCallingStreamingTransformation:
    """Test streaming function call transformations."""

    def setup_method(self):
        self.adapter = ResponseAdapter()

    @pytest.mark.asyncio
    async def test_streaming_with_tool_calls(self):
        """Test streaming response with function calls."""
        # Mock SSE stream with tool call events
        sse_chunks = [
            # Initial response setup
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "I\'ll help you"}]}]}\n\n',
            # Tool call start
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "tool_call", "id": "call_123", "function": {"name": "get_weather"}}]}]}\n\n',
            # Tool call arguments (partial)
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "tool_call", "id": "call_123", "function": {"arguments": "{\\"location\\""}}]}]}\n\n',
            # Tool call arguments (continued)
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "tool_call", "id": "call_123", "function": {"arguments": ": \\"SF\\"}"}}]}]}\n\n',
            # Completion
            b'event: response.completed\ndata: {"response": {"id": "resp_123", "model": "gpt-5", "usage": {"input_tokens": 20, "output_tokens": 15}}}\n\n',
        ]

        # Create async iterator from chunks
        async def mock_stream():
            for chunk in sse_chunks:
                yield chunk

        # Process the stream
        chunks = []
        async for chunk in self.adapter.stream_response_to_chat(mock_stream()):
            chunks.append(chunk)

        # Verify we got the expected chunks
        assert len(chunks) > 0

        # Check for role chunk
        role_chunks = [
            c
            for c in chunks
            if c.get("choices", [{}])[0].get("delta", {}).get("role") == "assistant"
        ]
        assert len(role_chunks) >= 1

        # Check for content chunks
        content_chunks = [
            c for c in chunks if "content" in c.get("choices", [{}])[0].get("delta", {})
        ]
        assert len(content_chunks) >= 1

        # Check for tool call chunks
        tool_call_chunks = [
            c
            for c in chunks
            if "tool_calls" in c.get("choices", [{}])[0].get("delta", {})
        ]
        assert len(tool_call_chunks) >= 1

        # Check final completion chunk
        completion_chunks = [
            c
            for c in chunks
            if c.get("choices", [{}])[0].get("finish_reason") == "tool_calls"
        ]
        assert len(completion_chunks) == 1

        # Verify usage information in final chunk
        final_chunk = completion_chunks[0]
        assert "usage" in final_chunk
        assert final_chunk["usage"]["prompt_tokens"] == 20
        assert final_chunk["usage"]["completion_tokens"] == 15

    @pytest.mark.asyncio
    async def test_streaming_without_tool_calls(self):
        """Test streaming response without function calls."""
        sse_chunks = [
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello there"}]}]}\n\n',
            b'event: response.output.delta\ndata: {"output": [{"type": "message", "content": [{"type": "output_text", "text": "!"}]}]}\n\n',
            b'event: response.completed\ndata: {"response": {"id": "resp_456", "model": "gpt-5"}}\n\n',
        ]

        async def mock_stream():
            for chunk in sse_chunks:
                yield chunk

        chunks = []
        async for chunk in self.adapter.stream_response_to_chat(mock_stream()):
            chunks.append(chunk)

        # Should have content chunks and completion with stop reason
        completion_chunks = [
            c
            for c in chunks
            if c.get("choices", [{}])[0].get("finish_reason") == "stop"
        ]
        assert len(completion_chunks) == 1


class TestToolCallUtilities:
    """Test utility methods for tool call processing."""

    def setup_method(self):
        self.adapter = ResponseAdapter()

    def test_convert_tools_to_response_api(self):
        """Test the _convert_tools_to_response_api utility method."""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Perform calculations",
                    "parameters": {
                        "type": "object",
                        "properties": {"expression": {"type": "string"}},
                    },
                },
            }
        ]

        response_tools = self.adapter._convert_tools_to_response_api(tools)

        assert len(response_tools) == 1
        tool = response_tools[0]
        assert isinstance(tool, ResponseTool)
        assert tool.type == "function"
        assert tool.function.name == "calculator"
        assert tool.function.description == "Perform calculations"

    def test_convert_tool_choice_to_response_api(self):
        """Test the _convert_tool_choice_to_response_api utility method."""
        # Test string choices
        assert self.adapter._convert_tool_choice_to_response_api("auto") == "auto"
        assert self.adapter._convert_tool_choice_to_response_api("none") == "none"
        assert (
            self.adapter._convert_tool_choice_to_response_api("required") == "required"
        )
        assert self.adapter._convert_tool_choice_to_response_api(None) == "auto"

        # Test function choice
        func_choice = {"type": "function", "function": {"name": "my_func"}}
        result = self.adapter._convert_tool_choice_to_response_api(func_choice)
        assert isinstance(result, ResponseToolChoice)
        assert result.type == "function"
        assert result.function["name"] == "my_func"

    def test_extract_tool_calls_from_output(self):
        """Test the _extract_tool_calls_from_output utility method."""
        output = [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Processing..."},
                    {
                        "type": "tool_call",
                        "id": "call_extract_test",
                        "function": {
                            "name": "extract_func",
                            "arguments": '{"param": "value"}',
                        },
                    },
                ],
            }
        ]

        tool_calls = self.adapter._extract_tool_calls_from_output(output)

        assert tool_calls is not None
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_extract_test"
        assert tool_calls[0].function.name == "extract_func"
        assert tool_calls[0].function.arguments == '{"param": "value"}'

    def test_extract_tool_calls_from_empty_output(self):
        """Test extracting tool calls from output without any tool calls."""
        output = [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "No tools here"}],
            }
        ]

        tool_calls = self.adapter._extract_tool_calls_from_output(output)
        assert tool_calls is None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
