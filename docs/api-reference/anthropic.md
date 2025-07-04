# Anthropic API Endpoints

Complete reference for Anthropic-compatible endpoints in your local Claude Code Proxy.

## Base URL

```
http://localhost:8000/v1
```

All Anthropic-compatible endpoints are prefixed with `/v1/`.

## Authentication

No API key required - uses your existing Claude CLI authentication automatically.

## Chat Completions

### POST /v1/chat/completions

Create a chat completion using Anthropic format.

#### Request

```http
POST /v1/chat/completions
Content-Type: application/json
```

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
| `model` | string | Yes | Claude model to use |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | Yes | Maximum tokens to generate |
| `temperature` | float | No | Sampling temperature (0.0-1.0) |
| `top_p` | float | No | Nucleus sampling parameter |
| `top_k` | integer | No | Top-k sampling parameter |
| `stream` | boolean | No | Enable streaming responses |
| `stop_sequences` | array | No | Stop sequences for completion |
| `system` | string | No | System prompt |

#### Response

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
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

#### Streaming Response

When `stream: true`, responses are sent as Server-Sent Events:

```
data: {"type": "message_start", "message": {...}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}

data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}}

data: [DONE]
```

## Models

### GET /v1/models

List available Claude models for your subscription.

#### Request

```http
GET /v1/models
```

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

## Error Responses

### Error Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid request parameters",
    "details": {
      "field": "model",
      "issue": "Model not found"
    }
  }
}
```

### Common Error Types

| Error Type | Status Code | Description |
|------------|-------------|-------------|
| `invalid_request_error` | 400 | Request validation failed |
| `authentication_error` | 401 | Authentication failed |
| `permission_error` | 403 | Permission denied |
| `not_found_error` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `internal_server_error` | 500 | Internal server error |
| `service_unavailable_error` | 503 | Service unavailable |

## Usage Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)

print(response.json())
```

### Python with Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy-key"  # Required but not used
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.content[0].text)
```

### curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming with curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 500,
    "stream": true
  }'
```