# API Reference

## Overview

The Claude Code Proxy API Server provides two compatible API interfaces:
- **Anthropic API Format** (`/v1/`) - Direct Anthropic API compatibility
- **OpenAI API Format** (`/openai/v1/`) - OpenAI API compatibility with format translation

Both interfaces support the same core functionality with different request/response formats.

## Base URL

```
http://localhost:8000
```

## Authentication

The proxy uses Claude CLI authentication. No API keys are required in requests as authentication is handled automatically through the Claude Code SDK.

## Common Headers

### Request Headers
```http
Content-Type: application/json
Accept: application/json
```

### Response Headers
```http
Content-Type: application/json
Access-Control-Allow-Origin: *
Access-Control-Allow-Headers: *
```

## Health Check

### GET /health

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "claude-proxy"
}
```

## Anthropic API Format (v1/)

### POST /v1/chat/completions

Create a chat completion using Anthropic API format.

#### Request Body

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "claude-3-5-sonnet-20241022") |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | No | Maximum tokens in response (default: 1000) |
| `temperature` | number | No | Sampling temperature 0.0-1.0 (default: 0.7) |
| `top_p` | number | No | Top-p sampling (default: 1.0) |
| `stream` | boolean | No | Enable streaming response (default: false) |
| `max_thinking_tokens` | integer | No | Maximum thinking tokens for reasoning |

#### Message Format

```json
{
  "role": "user|assistant|system",
  "content": "message content"
}
```

**Message Content Types:**

1. **Text Content**
```json
{
  "role": "user",
  "content": "Simple text message"
}
```

2. **Multi-modal Content**
```json
{
  "role": "user",
  "content": [
    {
      "type": "text",
      "text": "What's in this image?"
    },
    {
      "type": "image",
      "source": {
        "type": "base64",
        "media_type": "image/jpeg",
        "data": "base64_encoded_image_data"
      }
    }
  ]
}
```

#### Response (Non-streaming)

```json
{
  "id": "msg_123456789",
  "type": "message",
  "role": "assistant",
  "model": "claude-3-5-sonnet-20241022",
  "content": [
    {
      "type": "text",
      "text": "Hello! I'm doing well, thank you for asking."
    }
  ],
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 15,
    "output_tokens": 25
  }
}
```

#### Response (Streaming)

Streaming responses use Server-Sent Events format:

```
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"type": "message_start", "message": {"id": "msg_123", "model": "claude-3-5-sonnet-20241022"}}

data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello!"}}

data: {"type": "content_block_stop", "index": 0}

data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}}

data: {"type": "message_stop"}

data: [DONE]
```

### GET /v1/models

List available Claude models.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-3-5-sonnet-20241022",
      "object": "model",
      "created": 1677649963,
      "owned_by": "anthropic"
    },
    {
      "id": "claude-3-5-haiku-20241022",
      "object": "model",
      "created": 1677649963,
      "owned_by": "anthropic"
    }
  ]
}
```

## OpenAI API Format (openai/v1/)

### POST /openai/v1/chat/completions

Create a chat completion using OpenAI API format.

#### Request Body

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | No | Maximum tokens in response |
| `temperature` | number | No | Sampling temperature 0.0-2.0 |
| `top_p` | number | No | Top-p sampling |
| `frequency_penalty` | number | No | Frequency penalty (mapped to Claude parameters) |
| `presence_penalty` | number | No | Presence penalty (mapped to Claude parameters) |
| `stream` | boolean | No | Enable streaming response |
| `stop` | string/array | No | Stop sequences |
| `n` | integer | No | Number of completions (always 1 for Claude) |

#### Message Format

```json
{
  "role": "system|user|assistant",
  "content": "message content"
}
```

#### Response (Non-streaming)

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  }
}
```

#### Response (Streaming)

```
Content-Type: text/event-stream

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "claude-3-5-sonnet-20241022", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "claude-3-5-sonnet-20241022", "choices": [{"index": 0, "delta": {"content": "Hello"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "claude-3-5-sonnet-20241022", "choices": [{"index": 0, "delta": {"content": "!"}, "finish_reason": null}]}

data: {"id": "chatcmpl-123", "object": "chat.completion.chunk", "created": 1677652288, "model": "claude-3-5-sonnet-20241022", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}

data: [DONE]
```

### GET /openai/v1/models

List available models in OpenAI format.

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-3-5-sonnet-20241022",
      "object": "model",
      "created": 1677649963,
      "owned_by": "anthropic"
    }
  ]
}
```

## Supported Models

| Model ID | Description | Context Length |
|----------|-------------|----------------|
| `claude-3-5-sonnet-20241022` | Latest Sonnet model | 200K tokens |
| `claude-3-5-haiku-20241022` | Latest Haiku model | 200K tokens |
| `claude-3-opus-20240229` | Opus model | 200K tokens |
| `claude-3-sonnet-20240229` | Sonnet model | 200K tokens |
| `claude-3-haiku-20240307` | Haiku model | 200K tokens |

## Error Responses

### Anthropic Format Error

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid request: missing required field 'model'"
  }
}
```

### OpenAI Format Error

```json
{
  "error": {
    "message": "Invalid request: missing required field 'model'",
    "type": "invalid_request_error",
    "param": "model",
    "code": "missing_field"
  }
}
```

## Error Types

| Error Type | HTTP Status | Description |
|------------|-------------|-------------|
| `invalid_request_error` | 400 | Request validation failed |
| `authentication_error` | 401 | Authentication failed |
| `permission_error` | 403 | Permission denied |
| `not_found_error` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `internal_server_error` | 500 | Internal server error |
| `service_unavailable_error` | 503 | Service unavailable |

## Rate Limiting

The proxy implements rate limiting to prevent abuse:

- **Default Limits**: 100 requests per minute per IP
- **Headers**: Rate limit information in response headers
- **Retry**: Implement exponential backoff for rate limited requests

### Rate Limit Headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1640995200
```

## Usage Examples

### cURL Examples

#### Anthropic Format

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 1000
  }'
```

#### OpenAI Format

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Hello, how are you?"
      }
    ],
    "max_tokens": 1000
  }'
```

#### Streaming Request

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Write a short story"
      }
    ],
    "stream": true
  }'
```

### Python Examples

#### Using requests library

```python
import requests

# Anthropic format
response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [
            {
                "role": "user",
                "content": "Hello, how are you?"
            }
        ],
        "max_tokens": 1000
    }
)

data = response.json()
print(data["content"][0]["text"])
```

#### Using OpenAI Python library

```python
from openai import OpenAI

# Point to your local proxy
client = OpenAI(
    api_key="dummy-key",  # Not used but required
    base_url="http://localhost:8000/openai/v1"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ],
    max_tokens=1000
)

print(response.choices[0].message.content)
```

#### Streaming with OpenAI library

```python
from openai import OpenAI

client = OpenAI(
    api_key="dummy-key",
    base_url="http://localhost:8000/openai/v1"
)

stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {
            "role": "user",
            "content": "Write a short story"
        }
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

## WebSocket Support

Currently, the proxy does not support WebSocket connections. All communication is handled via HTTP requests with optional streaming responses using Server-Sent Events.

## Content Type Support

### Supported Content Types

- **Text**: Plain text messages
- **Images**: Base64-encoded images (JPEG, PNG, GIF, WebP)
- **Documents**: Text-based documents

### Image Format

```json
{
  "type": "image",
  "source": {
    "type": "base64",
    "media_type": "image/jpeg",
    "data": "base64_encoded_image_data"
  }
}
```

## API Versioning

The proxy maintains compatibility with:
- **Anthropic API**: Version 1 (v1)
- **OpenAI API**: Version 1 (v1)

Version information is included in the URL path (`/v1/` or `/openai/v1/`).

## Performance Considerations

- **Streaming**: Use streaming for long responses to reduce latency
- **Connection Pooling**: Reuse connections when possible
- **Timeout Handling**: Implement appropriate timeout handling
- **Error Retry**: Use exponential backoff for transient errors

## Security Notes

- **Authentication**: Handled automatically via Claude CLI
- **Rate Limiting**: Built-in protection against abuse
- **Input Validation**: All requests are validated before processing
- **Error Sanitization**: Error messages are sanitized to prevent information leakage