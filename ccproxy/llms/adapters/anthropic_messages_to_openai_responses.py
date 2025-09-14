from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any

from pydantic import BaseModel

from ccproxy.llms.adapters.base import BaseAPIAdapter
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.openai import models as openai_models


ResponseStreamEvent = (
    openai_models.ResponseCreatedEvent
    | openai_models.ResponseInProgressEvent
    | openai_models.ResponseCompletedEvent
    | openai_models.ResponseOutputTextDeltaEvent
    | openai_models.ResponseFunctionCallArgumentsDoneEvent
    | openai_models.ResponseRefusalDoneEvent
)


class AnthropicMessagesToOpenAIResponsesAdapter(
    BaseAPIAdapter[
        anthropic_models.CreateMessageRequest,
        anthropic_models.MessageResponse,
        anthropic_models.MessageStreamEvent,
    ]
):
    """Anthropic Messages ↔ OpenAI Responses adapter (request + response subset).

    Implemented
    - adapt_request: Anthropic → OpenAI Responses request
      - Maps `model`, `stream`, `max_tokens` → `max_output_tokens`
      - Maps last user message text into a single Responses `input` message
      - Maps custom tools → function tools
      - Maps tool_choice (auto/any/tool/none) and `parallel_tool_calls`
      - Maps `system` into `instructions` when present
    - adapt_response: Anthropic → OpenAI Responses response
      - Serializes `thinking` blocks into a single OutputTextContent using
        <thinking signature="…">…</thinking> XML followed by visible text
      - Maps ToolUseBlock into content dicts with type "tool_use"
      - Maps usage fields into OpenAI ResponseUsage

    TODO
    - Expand request mapping to include multiple turns and multimodal content
    - Include richer content types beyond text and tool use
    - Consider mapping stop reasons to status/incomplete details
    """

    def __init__(self) -> None:
        super().__init__(name="anthropic_messages_to_openai_responses")

    # Strongly-typed methods
    async def adapt_request(self, request: BaseModel) -> BaseModel:
        """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest."""
        if not isinstance(request, anthropic_models.CreateMessageRequest):
            raise ValueError(f"Expected CreateMessageRequest, got {type(request)}")

        return await self._convert_request(request)

    async def adapt_response(self, response: BaseModel) -> BaseModel:
        """Convert Anthropic MessageResponse to OpenAI ResponseObject."""
        if not isinstance(response, anthropic_models.MessageResponse):
            raise ValueError(f"Expected MessageResponse, got {type(response)}")

        return await self._convert_response(response)

    def adapt_stream(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[ResponseStreamEvent, None]:
        """Convert Anthropic MessageStreamEvent stream to OpenAI Response stream events."""
        return self._convert_stream(stream)

    async def _convert_stream(
        self, stream: AsyncIterator[anthropic_models.MessageStreamEvent]
    ) -> AsyncGenerator[ResponseStreamEvent, None]:
        item_id = "msg_stream"
        output_index = 0
        content_index = 0
        model_id = ""
        sequence_counter = 0

        async for evt in stream:
            if not hasattr(evt, "type"):
                continue

            sequence_counter += 1

            if evt.type == "message_start":
                model_id = evt.message.model or ""
                yield openai_models.ResponseCreatedEvent(
                    type="response.created",
                    sequence_number=sequence_counter,
                    response=openai_models.ResponseObject(
                        id=evt.message.id,
                        object="response",
                        created_at=0,
                        status="in_progress",
                        model=model_id,
                        output=[],
                        parallel_tool_calls=False,
                    ),
                )

                # Handle pre-filled content like thinking blocks
                for block in evt.message.content:
                    if block.type == "thinking":
                        sequence_counter += 1
                        thinking = block.thinking or ""
                        signature = block.signature
                        sig_attr = f' signature="{signature}"' if signature else ""
                        thinking_xml = f"<thinking{sig_attr}>{thinking}</thinking>"
                        yield openai_models.ResponseOutputTextDeltaEvent(
                            type="response.output_text.delta",
                            sequence_number=sequence_counter,
                            item_id=item_id,
                            output_index=output_index,
                            content_index=content_index,
                            delta=thinking_xml,
                        )

            elif evt.type == "content_block_start":
                if evt.content_block.type == "tool_use":
                    tool_input = evt.content_block.input or {}
                    try:
                        import json

                        args_str = json.dumps(tool_input, separators=(",", ":"))
                    except Exception:
                        args_str = str(tool_input)

                    yield openai_models.ResponseFunctionCallArgumentsDoneEvent(
                        type="response.function_call_arguments.done",
                        sequence_number=sequence_counter,
                        item_id=item_id,
                        output_index=output_index,
                        arguments=args_str,
                    )

            elif evt.type == "content_block_delta":
                text = evt.delta.text
                if text:
                    yield openai_models.ResponseOutputTextDeltaEvent(
                        type="response.output_text.delta",
                        sequence_number=sequence_counter,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        delta=text,
                    )

            elif evt.type == "message_delta":
                yield openai_models.ResponseInProgressEvent(
                    type="response.in_progress",
                    sequence_number=sequence_counter,
                    response=openai_models.ResponseObject(
                        id="",
                        object="response",
                        created_at=0,
                        status="in_progress",
                        model=model_id,
                        output=[],
                        parallel_tool_calls=False,
                        usage=openai_models.ResponseUsage(
                            input_tokens=evt.usage.input_tokens,
                            output_tokens=evt.usage.output_tokens,
                            total_tokens=evt.usage.input_tokens
                            + evt.usage.output_tokens,
                            input_tokens_details=openai_models.InputTokensDetails(
                                cached_tokens=0
                            ),
                            output_tokens_details=openai_models.OutputTokensDetails(
                                reasoning_tokens=0
                            ),
                        ),
                    ),
                )
                if evt.delta.stop_reason == "refusal":
                    sequence_counter += 1
                    yield openai_models.ResponseRefusalDoneEvent(
                        type="response.refusal.done",
                        sequence_number=sequence_counter,
                        item_id=item_id,
                        output_index=output_index,
                        content_index=content_index,
                        refusal="refused",
                    )

            elif evt.type == "message_stop":
                yield openai_models.ResponseCompletedEvent(
                    type="response.completed",
                    sequence_number=sequence_counter,
                    response=openai_models.ResponseObject(
                        id="",
                        object="response",
                        created_at=0,
                        status="completed",
                        model=model_id,
                        output=[],
                        parallel_tool_calls=False,
                    ),
                )
                break

    async def adapt_error(self, error: BaseModel) -> BaseModel:
        """Convert Anthropic error to OpenAI error format."""
        from ccproxy.llms.adapters.mapping import convert_anthropic_error_to_openai

        return convert_anthropic_error_to_openai(error)

    # Implementation methods
    async def _convert_request(
        self, request: anthropic_models.CreateMessageRequest
    ) -> openai_models.ResponseRequest:
        """Convert Anthropic CreateMessageRequest to OpenAI ResponseRequest using typed models."""
        # Build OpenAI Responses request payload
        payload_data: dict[str, Any] = {
            "model": request.model,
        }

        if request.max_tokens is not None:
            payload_data["max_output_tokens"] = int(request.max_tokens)
        if request.stream:
            payload_data["stream"] = True

        # Map system to instructions if present
        if request.system:
            if isinstance(request.system, str):
                payload_data["instructions"] = request.system
            else:
                payload_data["instructions"] = "".join(
                    block.text for block in request.system
                )

        # Map last user message text to Responses input
        last_user_text: str | None = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                if isinstance(msg.content, str):
                    last_user_text = msg.content
                elif isinstance(msg.content, list):
                    texts: list[str] = []
                    for block in msg.content:
                        # Support raw dicts and models
                        if isinstance(block, dict):
                            if block.get("type") == "text" and isinstance(
                                block.get("text"), str
                            ):
                                texts.append(block.get("text") or "")
                        else:
                            # Type guard for TextBlock
                            if (
                                getattr(block, "type", None) == "text"
                                and hasattr(block, "text")
                                and isinstance(getattr(block, "text", None), str)
                            ):
                                texts.append(block.text or "")
                    if texts:
                        last_user_text = " ".join(texts)
                break

        if last_user_text:
            payload_data["input"] = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": last_user_text},
                    ],
                }
            ]

        # Tools mapping (custom tools -> function tools)
        if request.tools:
            tools: list[dict[str, Any]] = []
            for tool in request.tools:
                if isinstance(tool, anthropic_models.Tool):
                    tools.append(
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            },
                        }
                    )
            if tools:
                payload_data["tools"] = tools

        # tool_choice mapping (+ parallel control)
        tc = request.tool_choice
        if tc is not None:
            tc_type = getattr(tc, "type", None)
            if tc_type == "none":
                payload_data["tool_choice"] = "none"
            elif tc_type == "auto":
                payload_data["tool_choice"] = "auto"
            elif tc_type == "any":
                payload_data["tool_choice"] = "required"
            elif tc_type == "tool":
                name = getattr(tc, "name", None)
                if name:
                    payload_data["tool_choice"] = {
                        "type": "function",
                        "function": {"name": name},
                    }
            disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
            if isinstance(disable_parallel, bool):
                payload_data["parallel_tool_calls"] = not disable_parallel

        # Validate
        return openai_models.ResponseRequest.model_validate(payload_data)

    async def _convert_response(
        self, response: anthropic_models.MessageResponse
    ) -> openai_models.ResponseObject:
        """Convert Anthropic MessageResponse to OpenAI ResponseObject using typed models."""
        # Aggregate thinking blocks (serialized) and text blocks into a single
        # OutputTextContent item, and include tool_use blocks as-is
        text_parts: list[str] = []
        other_contents: list[dict[str, Any]] = []
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "thinking":
                thinking = getattr(block, "thinking", None) or ""
                signature = getattr(block, "signature", None)
                sig_attr = (
                    f' signature="{signature}"'
                    if isinstance(signature, str) and signature
                    else ""
                )
                text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
            elif btype == "tool_use":
                other_contents.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", "tool_1"),
                        "name": getattr(block, "name", "function"),
                        # Prefer `input` -> `arguments` for Responses format
                        "arguments": getattr(block, "input", {}) or {},
                    }
                )

        msg_contents: list[dict[str, Any]] = []
        if text_parts:
            msg_contents.append(
                openai_models.OutputTextContent(
                    type="output_text", text="".join(text_parts)
                ).model_dump()
            )
        msg_contents.extend(other_contents)

        # Usage mapping
        usage = response.usage
        input_tokens_details = openai_models.InputTokensDetails(
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)
        resp_usage = openai_models.ResponseUsage(
            input_tokens=int(usage.input_tokens or 0),
            input_tokens_details=input_tokens_details,
            output_tokens=int(usage.output_tokens or 0),
            output_tokens_details=output_tokens_details,
            total_tokens=int((usage.input_tokens or 0) + (usage.output_tokens or 0)),
        )

        response_object = openai_models.ResponseObject(
            id=response.id,
            object="response",
            created_at=0,
            status="completed",
            model=response.model,
            output=[
                openai_models.MessageOutput(
                    type="message",
                    id=response.id,
                    status="completed",
                    role="assistant",
                    content=msg_contents,  # type: ignore[arg-type]
                )
            ],
            parallel_tool_calls=False,
            usage=resp_usage,
        )
        return response_object

    async def _adapt_response_dict_impl(
        self, response: dict[str, Any]
    ) -> dict[str, Any]:
        """Implementation moved from adapt_response - works with dicts."""
        # Validate incoming as Anthropic MessageResponse (or minimally shaped dict)
        anth = anthropic_models.MessageResponse.model_validate(response)

        # Aggregate thinking blocks (serialized) and text blocks into a single
        # OutputTextContent item, and include tool_use blocks as-is
        text_parts: list[str] = []
        other_contents: list[dict[str, Any]] = []
        for block in anth.content:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "thinking":
                thinking = getattr(block, "thinking", None) or ""
                signature = getattr(block, "signature", None)
                sig_attr = (
                    f' signature="{signature}"'
                    if isinstance(signature, str) and signature
                    else ""
                )
                text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
            elif btype == "tool_use":
                other_contents.append(
                    {
                        "type": "tool_use",
                        "id": getattr(block, "id", "tool_1"),
                        "name": getattr(block, "name", "function"),
                        # Prefer `input` -> `arguments` for Responses format
                        "arguments": getattr(block, "input", {}) or {},
                    }
                )

        msg_contents: list[dict[str, Any]] = []
        if text_parts:
            msg_contents.append(
                openai_models.OutputTextContent(
                    type="output_text", text="".join(text_parts)
                ).model_dump()
            )
        msg_contents.extend(other_contents)

        # Usage mapping
        usage = anth.usage
        input_tokens_details = openai_models.InputTokensDetails(
            cached_tokens=int(getattr(usage, "cache_read_input_tokens", 0) or 0)
        )
        output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)
        resp_usage = openai_models.ResponseUsage(
            input_tokens=int(usage.input_tokens or 0),
            input_tokens_details=input_tokens_details,
            output_tokens=int(usage.output_tokens or 0),
            output_tokens_details=output_tokens_details,
            total_tokens=int((usage.input_tokens or 0) + (usage.output_tokens or 0)),
        )

        response_object = openai_models.ResponseObject(
            id=anth.id,
            object="response",
            created_at=0,
            status="completed",
            model=anth.model,
            output=[
                openai_models.MessageOutput(
                    type="message",
                    id=anth.id,
                    status="completed",
                    role="assistant",
                    content=msg_contents,  # type: ignore[arg-type]
                )
            ],
            parallel_tool_calls=False,
            usage=resp_usage,
        )
        return response_object.model_dump()
