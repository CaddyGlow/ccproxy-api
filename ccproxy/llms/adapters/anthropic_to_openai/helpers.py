import time
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Any, Literal, cast

from pydantic import BaseModel

from ccproxy.llms.adapters.shared.constants import (
    ANTHROPIC_TO_OPENAI_ERROR_TYPE,
    ANTHROPIC_TO_OPENAI_FINISH_REASON,
)
from ccproxy.llms.anthropic import models as anthropic_models
from ccproxy.llms.openai import models as openai_models


FinishReason = Literal["stop", "length", "tool_calls"]

ResponseStreamEvent = (
    openai_models.ResponseCreatedEvent
    | openai_models.ResponseInProgressEvent
    | openai_models.ResponseCompletedEvent
    | openai_models.ResponseOutputTextDeltaEvent
    | openai_models.ResponseFunctionCallArgumentsDoneEvent
    | openai_models.ResponseRefusalDoneEvent
)


def convert__anthropic_usage_to_openai_completion__usage(
    usage: anthropic_models.Usage,
) -> openai_models.CompletionUsage:
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_creation_tokens > 0 and cached_tokens == 0:
        cached_tokens = cache_creation_tokens

    prompt_tokens_details = openai_models.PromptTokensDetails(
        cached_tokens=cached_tokens, audio_tokens=0
    )
    completion_tokens_details = openai_models.CompletionTokensDetails(
        reasoning_tokens=0,
        audio_tokens=0,
        accepted_prediction_tokens=0,
        rejected_prediction_tokens=0,
    )

    return openai_models.CompletionUsage(
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        prompt_tokens_details=prompt_tokens_details,
        completion_tokens_details=completion_tokens_details,
    )


def convert__anthropic_usage_to_openai_responses__usage(
    usage: anthropic_models.Usage,
) -> openai_models.ResponseUsage:
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    cached_tokens = int(getattr(usage, "cache_read_input_tokens", 0) or 0)
    cache_creation_tokens = int(getattr(usage, "cache_creation_input_tokens", 0) or 0)
    if cache_creation_tokens > 0 and cached_tokens == 0:
        cached_tokens = cache_creation_tokens

    input_tokens_details = openai_models.InputTokensDetails(cached_tokens=cached_tokens)
    output_tokens_details = openai_models.OutputTokensDetails(reasoning_tokens=0)

    return openai_models.ResponseUsage(
        input_tokens=input_tokens,
        input_tokens_details=input_tokens_details,
        output_tokens=output_tokens,
        output_tokens_details=output_tokens_details,
        total_tokens=input_tokens + output_tokens,
    )


# Error helpers migrated from ccproxy.llms.adapters.shared.errors


def convert__anthropic_to_openai__error(error: BaseModel) -> BaseModel:
    """Convert an Anthropic error payload to the OpenAI envelope."""
    from ccproxy.llms.anthropic.models import ErrorResponse as AnthropicErrorResponse
    from ccproxy.llms.openai.models import ErrorDetail
    from ccproxy.llms.openai.models import ErrorResponse as OpenAIErrorResponse

    if isinstance(error, AnthropicErrorResponse):
        anthropic_error = error.error
        error_message = anthropic_error.message
        anthropic_error_type = "api_error"
        if hasattr(anthropic_error, "type"):
            anthropic_error_type = anthropic_error.type

        openai_error_type = ANTHROPIC_TO_OPENAI_ERROR_TYPE.get(
            anthropic_error_type, "api_error"
        )

        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type=openai_error_type,
                code=None,
                param=None,
            )
        )

    if hasattr(error, "error") and hasattr(error.error, "message"):
        error_message = error.error.message
        return OpenAIErrorResponse(
            error=ErrorDetail(
                message=error_message,
                type="api_error",
                code=None,
                param=None,
            )
        )

    error_message = "Unknown error occurred"
    if hasattr(error, "message"):
        error_message = error.message
    elif hasattr(error, "model_dump"):
        error_dict = error.model_dump()
        if isinstance(error_dict, dict):
            error_message = error_dict.get("message", str(error_dict))

    return OpenAIErrorResponse(
        error=ErrorDetail(
            message=error_message,
            type="api_error",
            code=None,
            param=None,
        )
    )


async def convert__anthropic_message_to_openai_responses__stream(
    stream: AsyncIterator[anthropic_models.MessageStreamEvent],
) -> AsyncGenerator[ResponseStreamEvent, None]:
    item_id = "msg_stream"
    output_index = 0
    content_index = 0
    model_id = ""
    response_id = ""
    sequence_counter = 0

    async for evt in stream:
        if not hasattr(evt, "type"):
            continue

        sequence_counter += 1

        if evt.type == "message_start":
            model_id = evt.message.model or ""
            response_id = evt.message.id or ""
            yield openai_models.ResponseCreatedEvent(
                type="response.created",
                sequence_number=sequence_counter,
                response=openai_models.ResponseObject(
                    id=response_id,
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

                sequence_counter += 1
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
                sequence_counter += 1
                yield openai_models.ResponseOutputTextDeltaEvent(
                    type="response.output_text.delta",
                    sequence_number=sequence_counter,
                    item_id=item_id,
                    output_index=output_index,
                    content_index=content_index,
                    delta=text,
                )

        elif evt.type == "message_delta":
            sequence_counter += 1
            yield openai_models.ResponseInProgressEvent(
                type="response.in_progress",
                sequence_number=sequence_counter,
                response=openai_models.ResponseObject(
                    id=response_id,
                    object="response",
                    created_at=0,
                    status="in_progress",
                    model=model_id,
                    output=[],
                    parallel_tool_calls=False,
                    usage=cast(
                        openai_models.ResponseUsage,
                        convert__anthropic_usage_to_openai_responses__usage(evt.usage),
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
            sequence_counter += 1
            yield openai_models.ResponseCompletedEvent(
                type="response.completed",
                sequence_number=sequence_counter,
                response=openai_models.ResponseObject(
                    id=response_id,
                    object="response",
                    created_at=0,
                    status="completed",
                    model=model_id,
                    output=[],
                    parallel_tool_calls=False,
                ),
            )


def convert__anthropic_message_to_openai_responses__request(
    request: anthropic_models.CreateMessageRequest,
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

    # Always provide an input field matching ResponseRequest schema
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
    else:
        # Provide an empty input list if no user text detected to satisfy schema
        payload_data["input"] = []

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


async def convert__anthropic_message_to_openai_chat__stream(
    stream: AsyncIterator[anthropic_models.MessageStreamEvent],
) -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
    """Convert Anthropic stream to OpenAI stream using typed models."""

    async def generator() -> AsyncGenerator[openai_models.ChatCompletionChunk, None]:
        model_id = ""
        finish_reason: FinishReason = "stop"
        usage_prompt = 0
        usage_completion = 0

        async for evt in stream:
            if not hasattr(evt, "type"):
                continue

            if evt.type == "message_start":
                model_id = evt.message.model or ""
            elif evt.type == "content_block_delta":
                text = evt.delta.text
                if text:
                    yield openai_models.ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=text
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
            elif evt.type == "message_delta":
                if evt.delta.stop_reason:
                    finish_reason = cast(
                        FinishReason,
                        ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
                            evt.delta.stop_reason, "stop"
                        ),
                    )
                usage_prompt = evt.usage.input_tokens
                usage_completion = evt.usage.output_tokens
            elif evt.type == "message_stop":
                usage = None
                if usage_prompt or usage_completion:
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=usage_prompt,
                        completion_tokens=usage_completion,
                        total_tokens=usage_prompt + usage_completion,
                    )
                yield openai_models.ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
                    created=0,
                    model=model_id,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(),
                            finish_reason=finish_reason,
                        )
                    ],
                    usage=usage,
                )
                break

    return generator()


def convert__anthropic_message_to_openai_responses__response(
    response: anthropic_models.MessageResponse,
) -> openai_models.ResponseObject:
    """Convert Anthropic MessageResponse to an OpenAI ResponseObject."""
    text_parts: list[str] = []
    tool_contents: list[dict[str, Any]] = []
    for block in response.content:
        block_type = getattr(block, "type", None)
        if block_type == "text":
            text_parts.append(getattr(block, "text", ""))
        elif block_type == "thinking":
            thinking = getattr(block, "thinking", None) or ""
            signature = getattr(block, "signature", None)
            sig_attr = (
                f' signature="{signature}"'
                if isinstance(signature, str) and signature
                else ""
            )
            text_parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")
        elif block_type == "tool_use":
            tool_contents.append(
                {
                    "type": "tool_use",
                    "id": getattr(block, "id", "tool_1"),
                    "name": getattr(block, "name", "function"),
                    "arguments": getattr(block, "input", {}) or {},
                }
            )

    message_content: list[dict[str, Any]] = []
    if text_parts:
        message_content.append(
            openai_models.OutputTextContent(
                type="output_text",
                text="".join(text_parts),
            ).model_dump()
        )
    message_content.extend(tool_contents)

    usage_model = None
    if response.usage is not None:
        usage_model = cast(
            openai_models.ResponseUsage,
            convert__anthropic_usage_to_openai_responses__usage(response.usage),
        )

    return openai_models.ResponseObject(
        id=response.id,
        object="response",
        created_at=0,
        status="completed",
        model=response.model,
        output=[
            openai_models.MessageOutput(
                type="message",
                id=f"{response.id}_msg_0",
                status="completed",
                role="assistant",
                content=message_content,  # type: ignore[arg-type]
            )
        ],
        parallel_tool_calls=False,
        usage=usage_model,
    )


def convert__anthropic_message_to_openai_chat__request(
    request: anthropic_models.CreateMessageRequest,
) -> openai_models.ChatCompletionRequest:
    """Convert Anthropic CreateMessageRequest to OpenAI ChatCompletionRequest using typed models."""
    openai_messages: list[dict[str, Any]] = []
    # System prompt
    if request.system:
        if isinstance(request.system, str):
            sys_content = request.system
        else:
            sys_content = "".join(block.text for block in request.system)
        if sys_content:
            openai_messages.append({"role": "system", "content": sys_content})

    # User/assistant messages with text + data-url images
    for msg in request.messages:
        role = msg.role
        content = msg.content

        # Handle tool usage and results
        if role == "assistant" and isinstance(content, list):
            tool_calls = []
            text_parts = []
            for block in content:
                block_type = getattr(block, "type", None)
                if block_type == "tool_use":
                    # Type guard for ToolUseBlock
                    if (
                        hasattr(block, "id")
                        and hasattr(block, "name")
                        and hasattr(block, "input")
                    ):
                        tool_calls.append(
                            {
                                "id": block.id,
                                "type": "function",
                                "function": {
                                    "name": block.name,
                                    "arguments": str(block.input),
                                },
                            }
                        )
                elif block_type == "text":
                    # Type guard for TextBlock
                    if hasattr(block, "text"):
                        text_parts.append(block.text)
            if tool_calls:
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": tool_calls,
                }
                assistant_msg["content"] = " ".join(text_parts) if text_parts else None
                openai_messages.append(assistant_msg)
                continue
        elif role == "user" and isinstance(content, list):
            is_tool_result = any(
                getattr(b, "type", None) == "tool_result" for b in content
            )
            if is_tool_result:
                for block in content:
                    if getattr(block, "type", None) == "tool_result":
                        # Type guard for ToolResultBlock
                        if hasattr(block, "tool_use_id") and hasattr(block, "content"):
                            openai_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": block.tool_use_id,
                                    "content": str(block.content),
                                }
                            )
                continue

        if isinstance(content, list):
            parts: list[dict[str, Any]] = []
            text_accum: list[str] = []
            for block in content:
                # Support both raw dicts and Anthropic model instances
                if isinstance(block, dict):
                    btype = block.get("type")
                    if btype == "text" and isinstance(block.get("text"), str):
                        text_accum.append(block.get("text") or "")
                    elif btype == "image":
                        source = block.get("source") or {}
                        if (
                            isinstance(source, dict)
                            and source.get("type") == "base64"
                            and isinstance(source.get("media_type"), str)
                            and isinstance(source.get("data"), str)
                        ):
                            url = f"data:{source['media_type']};base64,{source['data']}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
                else:
                    # Pydantic models
                    btype = getattr(block, "type", None)
                    if (
                        btype == "text"
                        and hasattr(block, "text")
                        and isinstance(getattr(block, "text", None), str)
                    ):
                        text_accum.append(block.text or "")
                    elif btype == "image":
                        source = getattr(block, "source", None)
                        if (
                            source is not None
                            and getattr(source, "type", None) == "base64"
                            and isinstance(getattr(source, "media_type", None), str)
                            and isinstance(getattr(source, "data", None), str)
                        ):
                            url = f"data:{source.media_type};base64,{source.data}"
                            parts.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": url},
                                }
                            )
            if parts or len(text_accum) > 1:
                if text_accum:
                    parts.insert(0, {"type": "text", "text": " ".join(text_accum)})
                openai_messages.append({"role": role, "content": parts})
            else:
                openai_messages.append(
                    {"role": role, "content": (text_accum[0] if text_accum else "")}
                )
        else:
            openai_messages.append({"role": role, "content": content})

    # Tools mapping (custom tools -> function tools)
    tools: list[dict[str, Any]] = []
    if request.tools:
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

    params: dict[str, Any] = {
        "model": request.model,
        "messages": openai_messages,
        "max_completion_tokens": request.max_tokens,
    }
    if tools:
        params["tools"] = tools

    # tool_choice mapping
    tc = request.tool_choice
    if tc is not None:
        tc_type = getattr(tc, "type", None)
        if tc_type == "none":
            params["tool_choice"] = "none"
        elif tc_type == "auto":
            params["tool_choice"] = "auto"
        elif tc_type == "any":
            params["tool_choice"] = "required"
        elif tc_type == "tool":
            name = getattr(tc, "name", None)
            if name:
                params["tool_choice"] = {
                    "type": "function",
                    "function": {"name": name},
                }
        # parallel_tool_calls from disable_parallel_tool_use
        disable_parallel = getattr(tc, "disable_parallel_tool_use", None)
        if isinstance(disable_parallel, bool):
            params["parallel_tool_calls"] = not disable_parallel

    # Validate against OpenAI model
    return openai_models.ChatCompletionRequest.model_validate(params)


def convert__anthropic_message_to_openai_chat__response(
    response: anthropic_models.MessageResponse,
) -> openai_models.ChatCompletionResponse:
    """Convert Anthropic MessageResponse to an OpenAI ChatCompletionResponse."""
    content_blocks = response.content
    parts: list[str] = []
    for block in content_blocks:
        btype = getattr(block, "type", None)
        if btype == "text":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
        elif btype == "thinking":
            thinking = getattr(block, "thinking", None)
            signature = getattr(block, "signature", None)
            if isinstance(thinking, str):
                sig_attr = (
                    f' signature="{signature}"'
                    if isinstance(signature, str) and signature
                    else ""
                )
                parts.append(f"<thinking{sig_attr}>{thinking}</thinking>")

    content_text = "".join(parts)

    stop_reason = response.stop_reason
    finish_reason = ANTHROPIC_TO_OPENAI_FINISH_REASON.get(
        stop_reason or "end_turn", "stop"
    )

    usage_model = convert__anthropic_usage_to_openai_completion__usage(response.usage)

    payload = {
        "id": response.id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content_text},
                "finish_reason": finish_reason,
            }
        ],
        "created": int(time.time()),
        "model": response.model,
        "object": "chat.completion",
        "usage": usage_model.model_dump(),
    }
    return openai_models.ChatCompletionResponse.model_validate(payload)
