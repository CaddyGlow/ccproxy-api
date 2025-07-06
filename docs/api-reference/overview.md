# API Reference Overview

Claude Code Proxy API provides both Anthropic and OpenAI-compatible endpoints for seamless integration with Claude AI models.

## Base URLs

| API Format | Base URL | Description |
|------------|----------|-------------|
| Anthropic | `http://localhost:8000/v1` | Native Anthropic API format |
| OpenAI | `http://localhost:8000/openai/v1` | OpenAI-compatible format |

## Endpoint Summary

### Chat Completions

| Method | Anthropic Endpoint | OpenAI Endpoint | Description |
|--------|-------------------|-----------------|-------------|
| POST | `/v1/chat/completions` | `/openai/v1/chat/completions` | Create chat completion |

### Models

| Method | Anthropic Endpoint | OpenAI Endpoint | Description |
|--------|-------------------|-----------------|-------------|
| GET | `/v1/models` | `/openai/v1/models` | List available models |

### Health & Monitoring

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service health check |

## Authentication

Claude Code Proxy integrates with Claude CLI for authentication. Additionally, you can optionally secure the API endpoints with token authentication.

### Optional API Authentication

When `AUTH_TOKEN` is configured, the proxy automatically works with standard client libraries:

- **Anthropic SDK**: Sends API key as `x-api-key` header
- **OpenAI SDK**: Sends API key as `Authorization: Bearer` header

Both formats are accepted and use the same `AUTH_TOKEN` value. This means you can use the official SDKs without any modifications - just set the `api_key` parameter as usual.

### Request Headers

```http
Content-Type: application/json

# Optional authentication (if AUTH_TOKEN is configured)
x-api-key: your-auth-token
# OR
Authorization: Bearer your-auth-token
```

Note: When authentication is not configured, no authentication headers are required.

## Request/Response Format

### Anthropic Format

Follows the standard Anthropic API format:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "max_tokens": 1000
}
```

### OpenAI Format

Compatible with OpenAI's chat completions format:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello!"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

## Streaming

Both endpoints support streaming responses:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

## Error Handling

The API uses standard HTTP status codes and returns detailed error information:

### Error Response Format

```json
{
  "error": {
    "type": "validation_error",
    "message": "Invalid request parameters",
    "details": {
      "field": "model",
      "issue": "Model not found"
    }
  }
}
```

### Common Status Codes

| Code | Description | Common Causes |
|------|-------------|---------------|
| 200 | Success | Request completed successfully |
| 400 | Bad Request | Invalid request format or parameters |
| 401 | Unauthorized | Authentication failed |
| 404 | Not Found | Model or endpoint not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server or Claude CLI error |
| 503 | Service Unavailable | Claude CLI not available |

## Rate Limiting

The API includes built-in rate limiting to protect against abuse:

- **Default**: 60 requests per minute per IP
- **Burst**: 10 additional requests allowed
- **Headers**: Rate limit information included in response headers

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

## Content Types

### Supported Input Content Types
- `application/json` (required)

### Supported Output Content Types
- `application/json` (default)
- `text/event-stream` (for streaming responses)

## Models

Currently supported Claude models:

| Model ID | Description | Context Length |
|----------|-------------|----------------|
| `claude-3-5-sonnet-20241022` | Latest Claude 3.5 Sonnet | 200K tokens |
| `claude-3-7-sonnet-20250219` | Latest Claude 3.5 Haiku | 200K tokens |
| `claude-opus-4-20250514` | Claude 3 Opus | 200K tokens |

## API Compatibility

### Anthropic API Compatibility

The proxy is fully compatible with:
- Anthropic Python SDK
- Anthropic REST API
- All Anthropic message formats

### OpenAI API Compatibility

The proxy supports OpenAI-compatible:
- Chat completions endpoint
- Streaming responses
- Error response format
- Model listing

Note: Only chat completions are supported. Other OpenAI endpoints (completions, embeddings, etc.) are not available.

## Client Libraries

### Python

```python
# Using OpenAI client - works exactly like the official OpenAI API
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="your-auth-token"  # If AUTH_TOKEN is set, use it here. Otherwise any value works.
)

# That's it! The OpenAI SDK automatically handles the Bearer token
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)

# Using Anthropic client - works exactly like the official Anthropic API
from anthropic import Anthropic
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="your-auth-token"  # If AUTH_TOKEN is set, use it here. Otherwise any value works.
)

# That's it! The Anthropic SDK automatically handles the x-api-key header
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)

# Using requests directly
import requests
headers = {"Content-Type": "application/json"}

# With authentication (if configured)
headers["x-api-key"] = "your-auth-token"
# Or: headers["Authorization"] = "Bearer your-auth-token"

response = requests.post("http://localhost:8000/v1/chat/completions", json=data, headers=headers)
```

### JavaScript/Node.js

```javascript
// Using OpenAI SDK - works exactly like the official OpenAI API
import OpenAI from 'openai';
const client = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'your-auth-token',  // If AUTH_TOKEN is set, use it here. Otherwise any value works.
});

// That's it! The OpenAI SDK automatically handles the Bearer token
const response = await client.chat.completions.create({
  model: 'claude-sonnet-4-20250514',
  messages: [{ role: 'user', content: 'Hello!' }],
});

// Using fetch directly
const headers = { 'Content-Type': 'application/json' };

// With authentication (if configured)
headers['x-api-key'] = 'your-auth-token';
// Or: headers['Authorization'] = 'Bearer your-auth-token';

const response = await fetch('http://localhost:8000/v1/chat/completions', {
  method: 'POST',
  headers: headers,
  body: JSON.stringify(requestData)
});
```

### curl

```bash
# Anthropic format (without authentication)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'

# Anthropic format (with authentication)
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-auth-token" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'

# OpenAI format (with Bearer authentication)
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-auth-token" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'
```

## Next Steps

- [Anthropic Endpoints](anthropic.md) - Detailed Anthropic API documentation
- [OpenAI Endpoints](openai.md) - Detailed OpenAI API documentation  
- [Models](models.md) - Available models and their capabilities
- [Health Check](health.md) - Health monitoring endpoints
