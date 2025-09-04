"""Adapter for converting between OpenAI Chat Completions and Response API formats.

This adapter handles bidirectional conversion between:
- OpenAI Chat Completions API (used by most OpenAI clients)
- OpenAI Response API (used by Codex/ChatGPT backend)
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import structlog

from ccproxy.adapters.openai.models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChoice,
    OpenAIResponseMessage,
    OpenAIUsage,
    format_openai_tool_call,
)
from ccproxy.adapters.openai.response_models import (
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
from collections.abc import AsyncIterator


logger = structlog.get_logger(__name__)


class ResponseAdapter:
    """Adapter for OpenAI Response API format conversion."""

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
        # For Codex, we typically use gpt-5
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
        # Note: Response API always requires stream=true and store=false
        # Also, Response API doesn't support temperature and other OpenAI-specific parameters
        request_data = {
            "model": response_model,
            "instructions": instructions,
            "input": response_input,
            "stream": True,  # Always use streaming for Response API
            "tool_choice": tool_choice,
            "parallel_tool_calls": chat_dict.get("parallel_tool_calls", False),
            "reasoning": ResponseReasoning(effort="medium", summary="auto"),
            "store": False,  # Must be false for Response API
            # The following parameters are not supported by Response API:
            # temperature, max_output_tokens, top_p, frequency_penalty, presence_penalty
        }

        # Add tools if present
        if tools:
            request_data["tools"] = tools

        # Note: max_tool_calls is not supported by Response API
        # It will be filtered out during transformation

        request = ResponseRequest(**request_data)

        return request

    def _convert_tools_to_response_api(
        self, tools: list[dict[str, Any]]
    ) -> list[ResponseTool]:
        """Convert Chat Completions tools to Response API format.

        Args:
            tools: List of OpenAI Chat Completions tools

        Returns:
            List of Response API tools
        """
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
        """Convert Chat Completions tool_choice to Response API format.

        Args:
            tool_choice: OpenAI Chat Completions tool_choice

        Returns:
            Response API tool_choice
        """
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
        """Convert Response API response to Chat Completions format.

        Args:
            response_data: Response API response

        Returns:
            Chat Completions formatted response
        """
        # Extract the actual response data
        response_dict: dict[str, Any]
        if isinstance(response_data, ResponseCompleted):
            # Convert Pydantic model to dict
            response_dict = response_data.response.model_dump()
        else:  # isinstance(response_data, dict)
            if "response" in response_data:
                response_dict = response_data["response"]
            else:
                response_dict = response_data

        # Extract content from Response API output
        content = ""
        output = response_dict.get("output", [])
        # Look for message type output (skip reasoning)
        for output_item in output:
            if output_item.get("type") == "message":
                output_content = output_item.get("content", [])
                for content_block in output_content:
                    if content_block.get("type") in ["output_text", "text"]:
                        content += content_block.get("text", "")
                    # Skip tool_call blocks here - they're handled separately

        # Build Chat Completions response
        usage_data = response_dict.get("usage")
        converted_usage = self._convert_usage(usage_data) if usage_data else None

        # Extract tool calls from content
        tool_calls = self._extract_tool_calls_from_output(output)

        # Determine finish reason
        finish_reason = "tool_calls" if tool_calls else "stop"

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
        """Extract tool calls from Response API output.

        Args:
            output: Response API output array

        Returns:
            List of OpenAI-format tool calls or None
        """
        tool_calls = []

        for output_item in output:
            if output_item.get("type") == "message":
                output_content = output_item.get("content", [])
                for content_block in output_content:
                    if content_block.get("type") == "tool_call":
                        # Convert Response API tool call to Anthropic-style format
                        # then use existing format_openai_tool_call function
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

    async def stream_response_to_chat(
        self, response_stream: AsyncIterator[bytes]
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert Response API SSE stream to Chat Completions format.

        Args:
            response_stream: Async iterator of SSE bytes from Response API

        Yields:
            Chat Completions formatted streaming chunks
        """
        stream_id = f"chatcmpl_{uuid.uuid4().hex[:29]}"
        created = int(time.time())
        accumulated_content = ""
        buffer = ""

        # Tool call state tracking
        tool_calls_state: dict[str, dict[str, Any]] = {}
        role_sent = False

        logger.debug("response_adapter_stream_started", stream_id=stream_id)
        raw_chunk_count = 0
        event_count = 0

        async for chunk in response_stream:
            raw_chunk_count += 1
            chunk_size = len(chunk)
            logger.debug(
                "response_adapter_raw_chunk_received",
                chunk_number=raw_chunk_count,
                chunk_size=chunk_size,
                buffer_size_before=len(buffer),
            )

            # Add chunk to buffer
            buffer += chunk.decode("utf-8")

            # Process complete SSE events (separated by double newlines)
            while "\n\n" in buffer:
                event_str, buffer = buffer.split("\n\n", 1)
                event_count += 1

                # Parse the SSE event
                event_type = None
                event_data = None

                for line in event_str.strip().split("\n"):
                    if not line:
                        continue

                    if line.startswith("event:"):
                        event_type = line[6:].strip()
                    elif line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            logger.debug(
                                "response_adapter_done_marker_found",
                                event_number=event_count,
                            )
                            continue
                        try:
                            event_data = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug(
                                "response_adapter_sse_parse_failed",
                                data_preview=data_str[:100],
                                event_number=event_count,
                            )
                            continue

                # Process complete events
                if event_type and event_data:
                    logger.debug(
                        "response_adapter_sse_event_parsed",
                        event_type=event_type,
                        event_number=event_count,
                        has_output="output" in str(event_data),
                    )
                    if event_type in [
                        "response.output.delta",
                        "response.output_text.delta",
                    ]:
                        # Extract delta content
                        delta_content = ""

                        # Handle different event structures
                        if event_type == "response.output_text.delta":
                            # Direct text delta event
                            delta_content = event_data.get("delta", "")
                        else:
                            # Standard output delta with nested structure
                            output = event_data.get("output", [])
                            if output:
                                for output_item in output:
                                    if output_item.get("type") == "message":
                                        content_blocks = output_item.get("content", [])
                                        for block in content_blocks:
                                            if block.get("type") in [
                                                "output_text",
                                                "text",
                                            ]:
                                                delta_content += block.get("text", "")
                                            elif block.get("type") == "tool_call":
                                                # Handle tool call delta
                                                for (
                                                    chunk
                                                ) in self._process_tool_call_delta(
                                                    block,
                                                    tool_calls_state,
                                                    stream_id,
                                                    created,
                                                ):
                                                    yield chunk

                        # Send initial role chunk if not sent yet
                        if not role_sent:
                            yield {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": event_data.get("model", "gpt-5"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"role": "assistant"},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            role_sent = True

                        if delta_content:
                            accumulated_content += delta_content

                            logger.debug(
                                "response_adapter_yielding_content",
                                content_length=len(delta_content),
                                accumulated_length=len(accumulated_content),
                            )

                            # Create Chat Completions streaming chunk
                            yield {
                                "id": stream_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": event_data.get("model", "gpt-5"),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": delta_content},
                                        "finish_reason": None,
                                    }
                                ],
                            }

                    elif event_type == "response.completed":
                        # Final chunk with usage info
                        response = event_data.get("response", {})
                        usage = response.get("usage")

                        logger.debug(
                            "response_adapter_stream_completed",
                            total_content_length=len(accumulated_content),
                            has_usage=usage is not None,
                        )

                        # Determine finish reason based on tool calls
                        has_tool_calls = bool(tool_calls_state)
                        finish_reason = "tool_calls" if has_tool_calls else "stop"

                        chunk_data = {
                            "id": stream_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": response.get("model", "gpt-5"),
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": finish_reason,
                                }
                            ],
                        }

                        # Add usage if available
                        converted_usage = self._convert_usage(usage) if usage else None
                        if converted_usage:
                            chunk_data["usage"] = converted_usage.model_dump()

                        yield chunk_data

        logger.debug(
            "response_adapter_stream_finished",
            stream_id=stream_id,
            total_raw_chunks=raw_chunk_count,
            total_events=event_count,
            final_buffer_size=len(buffer),
        )

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

    def _process_tool_call_delta(
        self,
        block: dict[str, Any],
        tool_calls_state: dict[str, dict[str, Any]],
        stream_id: str,
        created: int,
    ):
        """Process tool call delta events and yield streaming chunks.

        Args:
            block: Tool call content block from Response API
            tool_calls_state: State tracking for tool calls
            stream_id: Stream ID
            created: Timestamp

        Yields:
            Chat Completions streaming chunks for tool calls
        """
        tool_id = block.get("id", "")
        function_data = block.get("function", {})
        function_name = function_data.get("name")
        function_args = function_data.get("arguments", "")

        if tool_id not in tool_calls_state:
            # Start of new tool call
            tool_calls_state[tool_id] = {
                "id": tool_id,
                "name": function_name or "",
                "arguments": "",
            }

            # Send tool call start chunk if we have the name
            if function_name:
                yield {
                    "id": stream_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": "gpt-5",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": len(tool_calls_state) - 1,
                                        "id": tool_id,
                                        "type": "function",
                                        "function": {"name": function_name},
                                    }
                                ]
                            },
                            "finish_reason": None,
                        }
                    ],
                }
        else:
            # Update existing tool call name if provided
            if function_name and not tool_calls_state[tool_id]["name"]:
                tool_calls_state[tool_id]["name"] = function_name

        # Handle function arguments delta
        if function_args:
            tool_calls_state[tool_id]["arguments"] += function_args

            # Send arguments delta
            yield {
                "id": stream_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": "gpt-5",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": list(tool_calls_state.keys()).index(
                                        tool_id
                                    ),
                                    "function": {"arguments": function_args},
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }

    def _get_default_codex_instructions(self) -> str:
        """Get default Codex CLI instructions."""
        return (
            "You are a coding agent running in the Codex CLI, a terminal-based coding assistant. "
            "Codex CLI is an open source project led by OpenAI. You are expected to be precise, safe, and helpful.\n\n"
            "Your capabilities:\n"
            "- Receive user prompts and other context provided by the harness, such as files in the workspace.\n"
            "- Communicate with the user by streaming thinking & responses, and by making & updating plans.\n"
            "- Emit function calls to run terminal commands and apply patches. Depending on how this specific run is configured, "
            "you can request that these function calls be escalated to the user for approval before running. "
            'More on this in the "Sandbox and approvals" section.\n\n'
            "Within this context, Codex refers to the open-source agentic coding interface "
            "(not the old Codex language model built by OpenAI)."
        )
