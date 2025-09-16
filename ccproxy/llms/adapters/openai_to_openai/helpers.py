from ccproxy.llms.adapters.shared.usage import (
    convert_openai_completion_usage_to_openai_response_usage,
)
import ccproxy.llms.openai.models as openai_models
# import ccproxy.llms.anthropic.models as anthropic_models


async def convert__openai_response_to_openaichat__request(
    request: openai_models.ResponseRequest,
) -> openai_models.ChatCompletionRequest:
    system_message: str | None = request.instructions
    messages: list[dict[str, Any]] = []

    # Handle string input shortcut
    if isinstance(request.input, str):
        messages.append({"role": "user", "content": request.input})
    else:
        for item in request.input or []:
            role = getattr(item, "role", None) or "user"
            content_blocks = getattr(item, "content", [])
            text_parts: list[str] = []

            for part in content_blocks or []:
                if isinstance(part, dict):
                    if part.get("type") in {"input_text", "text"}:
                        text = part.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                else:
                    part_type = getattr(part, "type", None)
                    if part_type in {"input_text", "text"} and hasattr(part, "text"):
                        text_value = part.text
                        if isinstance(text_value, str):
                            text_parts.append(text_value)

            if role == "system":
                # Merge all system content into a single system message
                system_message = " ".join([p for p in text_parts if p])
            else:
                messages.append(
                    {
                        "role": role,
                        "content": " ".join([p for p in text_parts if p]) or None,
                    }
                )

    if system_message:
        messages.insert(0, {"role": "system", "content": system_message})

    # Provide a default user prompt if none extracted
    if not messages:
        messages.append({"role": "user", "content": ""})

    payload: dict[str, Any] = {
        "model": request.model or "gpt-4o-mini",
        "messages": messages,
    }

    if request.max_output_tokens is not None:
        payload["max_completion_tokens"] = request.max_output_tokens

    if request.stream is not None:
        payload["stream"] = request.stream

    if request.temperature is not None:
        payload["temperature"] = request.temperature

    if request.top_p is not None:
        payload["top_p"] = request.top_p

    if request.tools:
        payload["tools"] = request.tools

    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice

    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls

    return ChatCompletionRequest.model_validate(payload)


async def convert__openai_chat_to_openai_response__response(
    chat_response: ChatCompletionResponse,
) -> ResponseObject:
    content_text = ""
    if chat_response.choices:
        first_choice = chat_response.choices[0]
        if first_choice.message and first_choice.message.content:
            content_text = first_choice.message.content

    message_output = openai_models.MessageOutput(
        type="message",
        role="assistant",
        id=f"msg_{chat_response.id or 'unknown'}",
        status="completed",
        content=[
            openai_models.OutputTextContent(type="output_text", text=content_text)
        ],
    )

    usage: openai_models.ResponseUsage | None = None
    if chat_response.usage:
        usage = convert_openai_completion_usage_to_openai_response_usage(chat_response)

    return openai_models.ResponseObject(
        id=chat_response.id or "resp-unknown",
        object="response",
        created_at=int(time.time()),
        model=chat_response.model or "",
        status="completed",
        output=[message_output],
        parallel_tool_calls=False,
        usage=usage,
    )


def convert__openai_response_to_openai_chat__response(
    response: openai_models.ResponseObject,
) -> openai_models.ChatCompletionResponse:
    """Convert an OpenAI ResponseObject to a ChatCompletionResponse."""
    # Find first message output and aggregate output_text parts
    text_content = ""
    for item in response.output or []:
        if hasattr(item, "type") and item.type == "message":
            parts: list[str] = []
            for part in getattr(item, "content", []):
                if hasattr(part, "type") and part.type == "output_text":
                    if hasattr(part, "text") and isinstance(part.text, str):
                        parts.append(part.text)
                elif isinstance(part, dict) and part.get("type") == "output_text":
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            text_content = "".join(parts)
            break

    usage = None
    if response.usage:
        usage = convert_openai_response_usage_to_openai_completion_usage(response.usage)

    return openai_models.ChatCompletionResponse(
        id=response.id or "chatcmpl-resp",
        choices=[
            openai_models.Choice(
                index=0,
                message=openai_models.ResponseMessage(
                    role="assistant", content=text_content
                ),
                finish_reason="stop",
            )
        ],
        created=0,
        model=response.model or "",
        object="chat.completion",
        usage=usage
        or openai_models.CompletionUsage(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        ),
    )


def response_stream_to_chat_chunks(
    stream: AsyncIterator[AnyStreamEvent],
) -> AsyncGenerator[ChatCompletionChunk, None]:
    """Convert Response API stream events to ChatCompletionChunk events."""

    async def generator() -> AsyncGenerator[ChatCompletionChunk, None]:
        model_id = ""
        async for event_wrapper in stream:
            if hasattr(event_wrapper, "root"):
                evt = event_wrapper.root
            else:
                evt = event_wrapper  # type: ignore[arg-type]
            if not hasattr(evt, "type"):
                continue

            if evt.type == "response.created":
                model_id = getattr(evt.response, "model", "")
            elif evt.type == "response.output_text.delta":
                delta = getattr(evt, "delta", None) or ""
                if delta:
                    yield ChatCompletionChunk(
                        id="chatcmpl-stream",
                        object="chat.completion.chunk",
                        created=0,
                        model=model_id,
                        choices=[
                            openai_models.StreamingChoice(
                                index=0,
                                delta=openai_models.DeltaMessage(
                                    role="assistant", content=delta
                                ),
                                finish_reason=None,
                            )
                        ],
                    )
            elif evt.type in {
                "response.completed",
                "response.incomplete",
                "response.failed",
            }:
                usage = None
                response_obj = getattr(evt, "response", None)
                if response_obj and getattr(response_obj, "usage", None):
                    input_tokens, output_tokens, _ = safe_extract_usage_tokens(
                        response_obj.usage
                    )
                    usage = openai_models.CompletionUsage(
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                        total_tokens=input_tokens + output_tokens,
                    )
                yield ChatCompletionChunk(
                    id="chatcmpl-stream",
                    object="chat.completion.chunk",
                    created=0,
                    model=model_id,
                    choices=[
                        openai_models.StreamingChoice(
                            index=0,
                            delta=openai_models.DeltaMessage(),
                            finish_reason="stop",
                        )
                    ],
                    usage=usage,
                )
            break

    return generator()


async def convert__openai_chat_to_openai_response__request(
    self, request: openai_models.ChatCompletionRequest
) -> openai_models.ResponseRequest:
    """Convert ChatCompletionRequest to ResponseRequest using typed models."""
    model = request.model
    max_out = request.max_completion_tokens or request.max_tokens

    # Find the last user message
    user_text: str | None = None
    for msg in reversed(request.messages or []):
        if msg.role == "user":
            content = msg.content
            if isinstance(content, list):
                texts = [
                    part.text
                    for part in content
                    if hasattr(part, "type")
                    and part.type == "text"
                    and hasattr(part, "text")
                ]
                user_text = " ".join([t for t in texts if t])
            else:
                user_text = content
            break

    input_data = []
    if user_text:
        input_msg = {
            "type": "message",
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": user_text,
                }
            ],
        }
        input_data = [input_msg]

    payload_data: dict[str, Any] = {
        "model": model,
    }
    if max_out is not None:
        payload_data["max_output_tokens"] = int(max_out)
    if input_data:
        payload_data["input"] = input_data

    # Structured outputs: map Chat response_format to Responses text.format
    resp_fmt = request.response_format
    if resp_fmt is not None:
        if resp_fmt.type == "text":
            payload_data["text"] = {"format": {"type": "text"}}
        elif resp_fmt.type == "json_object":
            payload_data["text"] = {"format": {"type": "json_object"}}
        elif resp_fmt.type == "json_schema" and hasattr(resp_fmt, "json_schema"):
            js = resp_fmt.json_schema
            # Pass through name/schema/strict if provided
            fmt = {"type": "json_schema"}
            if js is not None:
                js_dict = js.model_dump() if hasattr(js, "model_dump") else js
                if js_dict is not None:
                    fmt.update(
                        {
                            k: v
                            for k, v in js_dict.items()
                            if k in {"name", "schema", "strict", "$defs", "description"}
                        }
                    )
            payload_data["text"] = {"format": fmt}

    if request.tools:
        payload_data["tools"] = [
            tool.model_dump() if hasattr(tool, "model_dump") else tool
            for tool in request.tools
        ]

    return openai_models.ResponseRequest.model_validate(payload_data)
