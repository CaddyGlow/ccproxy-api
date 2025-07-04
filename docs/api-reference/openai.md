# OpenAI API Endpoints

Complete reference for OpenAI-compatible endpoints in your local Claude Code Proxy.

## Base URL

```
http://localhost:8000/openai/v1
```

All OpenAI-compatible endpoints are prefixed with `/openai/v1/`.

## Authentication

No API key required - uses your existing Claude CLI authentication automatically.

**Note**: OpenAI client libraries require an API key parameter, but any dummy value will work (e.g., "dummy-key").

## Chat Completions

### POST /openai/v1/chat/completions

Create a chat completion using OpenAI format, automatically translated to Claude.

#### Request

```http
POST /openai/v1/chat/completions
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
| `max_tokens` | integer | No | Maximum tokens to generate |
| `temperature` | float | No | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | Nucleus sampling parameter |
| `stream` | boolean | No | Enable streaming responses |
| `stop` | array/string | No | Stop sequences for completion |
| `presence_penalty` | float | No | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | No | Frequency penalty (-2.0 to 2.0) |
| `user` | string | No | User identifier |

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

When `stream: true`, responses are sent as Server-Sent Events in OpenAI format:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Models

### GET /openai/v1/models

List available Claude models in OpenAI format.

#### Request

```http
GET /openai/v1/models
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

## Parameter Translation

The proxy automatically translates OpenAI parameters to Claude equivalents:

| OpenAI Parameter | Claude Parameter | Notes |
|------------------|------------------|-------|
| `max_tokens` | `max_tokens` | Direct mapping |
| `temperature` | `temperature` | Clamped to 0.0-1.0 range |
| `top_p` | `top_p` | Direct mapping |
| `stop` | `stop_sequences` | Converted to array format |
| `presence_penalty` | Not supported | Ignored with warning |
| `frequency_penalty` | Not supported | Ignored with warning |
| `user` | Not supported | Ignored |

## Error Responses

### Error Format

```json
{
  "error": {
    "message": "Invalid request parameters",
    "type": "invalid_request_error",
    "param": "model",
    "code": "invalid_model"
  }
}
```

### Common Error Codes

| Error Code | Status Code | Description |
|------------|-------------|-------------|
| `invalid_request_error` | 400 | Request validation failed |
| `invalid_api_key` | 401 | Authentication failed |
| `insufficient_quota` | 403 | Permission denied |
| `model_not_found` | 404 | Model not available |
| `rate_limit_exceeded` | 429 | Rate limit exceeded |
| `internal_server_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service unavailable |

## Usage Examples

### Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"  # Required but not used
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"
)

stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'dummy-key' // Required but not used
});

const completion = await openai.chat.completions.create({
  model: 'claude-3-5-sonnet-20241022',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 100
});

console.log(completion.choices[0].message.content);
```

### curl

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Migration from OpenAI

To migrate existing OpenAI-based applications:

1. **Change the base URL**:
   ```python
   # Before
   client = OpenAI(api_key="your-openai-key")
   
   # After
   client = OpenAI(
       base_url="http://localhost:8000/openai/v1",
       api_key="dummy-key"
   )
   ```

2. **Update model names** to Claude models:
   ```python
   # Before
   model="gpt-4-turbo"
   
   # After
   model="claude-3-5-sonnet-20241022"
   ```

3. **Remove unsupported parameters** (optional):
   - `presence_penalty`
   - `frequency_penalty`
   - `user`

That's it! Your existing OpenAI code should work without other changes.