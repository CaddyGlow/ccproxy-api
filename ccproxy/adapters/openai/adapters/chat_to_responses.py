"""Adapter for converting between OpenAI Chat Completions and Response API formats.

This adapter handles bidirectional conversion between:
- OpenAI Chat Completions API (used by most OpenAI clients)
- OpenAI Response API (used by Codex/ChatGPT backend)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal

import structlog

from ccproxy.adapters.base import BaseAPIAdapter
from ccproxy.adapters.openai.models.chat_completions import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIResponseMessage,
)
from ccproxy.adapters.openai.models.common import (
    OpenAIUsage,
    format_openai_tool_call,
)
from ccproxy.adapters.openai.models.responses import (
    ResponseCompleted,
    ResponseFunction,
    ResponseMessage,
    ResponseMessageContent,
    ResponseReasoning,
    ResponseRequest,
    ResponseTool,
    ResponseToolChoice,
    ResponseToolFunction,
)


logger = structlog.get_logger(__name__)


class ChatToResponsesAdapter(BaseAPIAdapter):
    """Adapter for converting between Chat Completions and Response API formats."""

    def __init__(self) -> None:
        super().__init__("chat_to_responses")

    async def adapt_request(self, request_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Chat Completions request to Response API format.

        Args:
            request_data: OpenAI Chat Completions request

        Returns:
            Response API formatted request
        """
        return self.chat_to_response_request(request_data).model_dump(exclude_none=True)

    async def adapt_response(self, response_data: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API response to Chat Completions format.

        Args:
            response_data: Response API response

        Returns:
            Chat Completions formatted response
        """
        result = self.response_to_chat_completion(response_data).model_dump(
            exclude_none=True
        )
        return dict(result)

    def adapt_stream(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert Response API streaming to Chat Completions streaming.

        Args:
            stream: Response API streaming events

        Yields:
            Chat Completions streaming chunks
        """
        return self._adapt_stream_impl(stream)

    def chat_to_response_request(
        self, chat_request: dict[str, Any] | OpenAIChatCompletionRequest
    ) -> ResponseRequest:
        """Convert Chat Completions request to Response API format.

        Args:
            chat_request: OpenAI Chat Completions request

        Returns:
            Response API formatted request
        """
        if isinstance(chat_request, OpenAIChatCompletionRequest):
            chat_dict = chat_request.model_dump()
        else:
            chat_dict = chat_request

        # Extract messages and convert to Response API format
        messages = chat_dict.get("messages", [])
        response_input = []
        instructions = None

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # System messages become instructions
            if role == "system":
                instructions = content
                continue

            # Convert user/assistant messages to Response API format
            content_blocks = []

            # Handle text content
            if isinstance(content, str) and content:
                content_blocks.append(
                    ResponseMessageContent(
                        type="input_text" if role == "user" else "output_text",
                        text=content,
                    )
                )
            elif isinstance(content, list):
                # Handle structured content (images, etc.)
                for block in content:
                    if block.get("type") == "text" and block.get("text"):
                        content_blocks.append(
                            ResponseMessageContent(
                                type="input_text" if role == "user" else "output_text",
                                text=block["text"],
                            )
                        )
            elif content:
                # Fallback for other content types
                content_blocks.append(
                    ResponseMessageContent(
                        type="input_text" if role == "user" else "output_text",
                        text=str(content),
                    )
                )

            # Handle tool calls for assistant messages
            if role == "assistant" and msg.get("tool_calls"):
                for tool_call in msg["tool_calls"]:
                    func = tool_call.get("function", {})
                    content_blocks.append(
                        ResponseMessageContent(
                            type="tool_call",
                            id=tool_call.get("id", ""),
                            function=ResponseFunction(
                                name=func.get("name", ""),
                                arguments=func.get("arguments", "{}"),
                            ),
                        )
                    )

            if content_blocks:
                response_msg = ResponseMessage(
                    type="message",
                    id=None,
                    role=role if role in ["user", "assistant"] else "user",
                    content=content_blocks,
                )
                response_input.append(response_msg)

        # Leave instructions field unset to let codex_transformers inject them
        # The backend validates instructions and needs the full Codex ones
        instructions = None
        # Actually, we need to not include the field at all if it's None
        # Otherwise the backend complains "Instructions are required"

        # Map model (Codex uses gpt-5)
        model = chat_dict.get("model", "gpt-4")
        response_model = (
            "gpt-5" if "codex" in model.lower() or "gpt-5" in model.lower() else model
        )

        # Convert tools if present
        tools = None
        if chat_dict.get("tools"):
            tools = self._convert_tools_to_response_api(chat_dict["tools"])

        # Convert tool_choice if present
        tool_choice = self._convert_tool_choice_to_response_api(
            chat_dict.get("tool_choice", "auto")
        )

        # Build Response API request
        request_data = {
            "model": response_model,
            "instructions": instructions,
            "input": response_input,
            "stream": True,  # Always use streaming for Response API
            "tool_choice": tool_choice,
            "parallel_tool_calls": chat_dict.get("parallel_tool_calls", False),
            "reasoning": ResponseReasoning(effort="medium", summary="auto"),
            "store": False,  # Must be false for Response API
        }

        # Add tools if present
        if tools:
            request_data["tools"] = tools

        request = ResponseRequest(**request_data)
        return request

    def _convert_tools_to_response_api(
        self, tools: list[dict[str, Any]]
    ) -> list[ResponseTool]:
        """Convert Chat Completions tools to Response API format."""
        response_tools = []

        for tool in tools:
            if tool.get("type") == "function":
                func = tool.get("function", {})
                response_tools.append(
                    ResponseTool(
                        type="function",
                        function=ResponseToolFunction(
                            name=func.get("name", ""),
                            description=func.get("description"),
                            parameters=func.get("parameters", {}),
                        ),
                    )
                )

        return response_tools

    def _convert_tool_choice_to_response_api(
        self, tool_choice: str | dict[str, Any] | None
    ) -> str | ResponseToolChoice:
        """Convert Chat Completions tool_choice to Response API format."""
        if tool_choice is None or tool_choice == "auto":
            return "auto"
        elif tool_choice == "none":
            return "none"
        elif tool_choice == "required":
            return "required"
        elif isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
            func = tool_choice.get("function", {})
            return ResponseToolChoice(
                type="function",
                function={"name": func.get("name", "")},
            )
        else:
            # Fallback to auto for unknown formats
            return "auto"

    def response_to_chat_completion(
        self, response_data: dict[str, Any] | ResponseCompleted
    ) -> OpenAIChatCompletionResponse:
        """Convert Response API response to Chat Completions format."""
        # Extract the actual response data
        response_dict: dict[str, Any]
        if isinstance(response_data, ResponseCompleted):
            response_dict = response_data.response.model_dump()
        else:
            if "response" in response_data:
                response_dict = response_data["response"]
            else:
                response_dict = response_data

        # Extract content from Response API output
        content = ""
        output = response_dict.get("output", [])
        for output_item in output:
            if output_item.get("type") == "message":
                output_content = output_item.get("content", [])
                for content_block in output_content:
                    if content_block.get("type") in ["output_text", "text"]:
                        content += content_block.get("text", "")

        # Build Chat Completions response
        usage_data = response_dict.get("usage")
        converted_usage = self._convert_usage(usage_data) if usage_data else None

        # Extract tool calls from content
        tool_calls = self._extract_tool_calls_from_output(output)

        # Determine finish reason
        finish_reason: Literal["stop", "length", "tool_calls", "content_filter"] = (
            "tool_calls" if tool_calls else "stop"
        )

        return OpenAIChatCompletionResponse(
            id=response_dict.get("id", f"resp_{uuid.uuid4().hex}"),
            object="chat.completion",
            created=response_dict.get("created_at", int(time.time())),
            model=response_dict.get("model", "gpt-5"),
            choices=[
                OpenAIChoice(
                    index=0,
                    message=OpenAIResponseMessage(
                        role="assistant",
                        content=content or None,
                        tool_calls=tool_calls if tool_calls else None,
                    ),
                    finish_reason=finish_reason,
                )
            ],
            usage=converted_usage,
            system_fingerprint=response_dict.get("safety_identifier"),
        )

    def _extract_tool_calls_from_output(
        self, output: list[dict[str, Any]]
    ) -> list[Any] | None:
        """Extract tool calls from Response API output."""
        tool_calls = []

        for output_item in output:
            if output_item.get("type") == "message":
                output_content = output_item.get("content", [])
                for content_block in output_content:
                    if content_block.get("type") == "tool_call":
                        func = content_block.get("function", {})

                        # Parse arguments JSON string to dict
                        arguments_str = func.get("arguments", "{}")
                        try:
                            if isinstance(arguments_str, str):
                                input_dict = json.loads(arguments_str)
                            else:
                                input_dict = arguments_str
                        except json.JSONDecodeError:
                            logger.warning(
                                "tool_arguments_parse_failed",
                                arguments=arguments_str[:200] + "..."
                                if len(str(arguments_str)) > 200
                                else str(arguments_str),
                                operation="extract_tool_calls_from_output",
                            )
                            input_dict = {}

                        # Create Anthropic-style tool use for conversion
                        anthropic_tool_use = {
                            "id": content_block.get("id", ""),
                            "name": func.get("name", ""),
                            "input": input_dict,
                        }

                        # Use existing conversion function
                        tool_calls.append(format_openai_tool_call(anthropic_tool_use))

        return tool_calls if tool_calls else None

    async def _adapt_stream_impl(
        self, stream: AsyncIterator[dict[str, Any]]
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Convert Response API SSE stream to Chat Completions format."""
        from collections import defaultdict

        stream_id = f"chatcmpl_{uuid.uuid4().hex[:29]}"
        created = int(time.time())
        role_sent = False
        event_counts: dict[str, int] = defaultdict(int)
        start_time = time.time()

        logger.info(
            "streaming_started",
            plugin="chat_to_responses",
            stream_id=stream_id,
        )

        async for event in stream:
            event_type = event.get("type", "unknown")
            event_counts[event_type] += 1

            # Send initial role chunk if not sent yet
            if not role_sent:
                yield {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                role_sent = True

            # Convert Response API events to ChatCompletion deltas
            chunk = self._convert_response_event_to_chat_delta(
                event, stream_id, created
            )
            if chunk:
                yield chunk

        # Log streaming summary
        duration = time.time() - start_time
        logger.info(
            "streaming_complete",
            plugin="chat_to_responses",
            stream_id=stream_id,
            duration_ms=round(duration * 1000, 2),
            event_summary=dict(event_counts),
            total_events=sum(event_counts.values()),
        )

    def _convert_response_event_to_chat_delta(
        self, event: dict[str, Any], stream_id: str, created: int
    ) -> dict[str, Any] | None:
        """Convert a Response API event to ChatCompletion delta format."""
        event_type = event.get("type", "")

        # Handle content deltas (main text output)
        if event_type == "response.output_text.delta":
            delta_text = event.get("delta", "")
            if delta_text:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_text},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle structured output deltas
        elif event_type == "response.output.delta":
            # Extract content from nested output structure
            output = event.get("output", [])
            delta_content = ""

            for output_item in output:
                if output_item.get("type") == "message":
                    content_blocks = output_item.get("content", [])
                    for block in content_blocks:
                        if block.get("type") in ["output_text", "text"]:
                            delta_content += block.get("text", "")

            if delta_content:
                return {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta_content},
                            "finish_reason": None,
                        }
                    ],
                }

        # Handle completion events
        elif event_type == "response.completed":
            response = event.get("response", {})
            usage = response.get("usage")

            chunk_data = {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": response.get("model", "gpt-5"),
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }

            # Add usage if available
            if usage:
                chunk_data["usage"] = {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }

            return chunk_data

        return None

    def _convert_usage(
        self, response_usage: dict[str, Any] | None
    ) -> OpenAIUsage | None:
        """Convert Response API usage to Chat Completions format."""
        if not response_usage:
            return None

        return OpenAIUsage(
            prompt_tokens=response_usage.get("input_tokens", 0),
            completion_tokens=response_usage.get("output_tokens", 0),
            total_tokens=response_usage.get("total_tokens", 0),
        )

    async def adapt_error(self, error: dict[str, Any]) -> dict[str, Any]:
        """Convert Response API error format to Chat Completions error format.

        Args:
            error: Response API error response

        Returns:
            Chat Completions error response
        """
        # Extract error details from Response API format
        response_error = error.get("error", {})
        error_type = response_error.get("type", "internal_server_error")
        error_message = response_error.get("message", "An error occurred")

        # Map Response API error types to OpenAI error types
        error_type_mapping = {
            "invalid_request_error": "invalid_request_error",
            "authentication_error": "invalid_request_error",
            "permission_error": "invalid_request_error",
            "not_found_error": "invalid_request_error",
            "rate_limit_error": "rate_limit_error",
            "internal_server_error": "internal_server_error",
            "overloaded_error": "server_error",
        }

        openai_error_type = error_type_mapping.get(error_type, "invalid_request_error")

        # Return OpenAI Chat Completions error format
        return {
            "error": {
                "message": error_message,
                "type": openai_error_type,
                "code": error_type,
            }
        }


__all__ = ["ChatToResponsesAdapter"]
