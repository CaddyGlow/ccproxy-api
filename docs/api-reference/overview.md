# API Reference Overview

Claude Code Proxy API provides both Anthropic and OpenAI-compatible endpoints for seamless integration with Claude AI models.

## Base URLs by Mode

| Mode | Anthropic Base URL | OpenAI Base URL | Authentication Support |
|------|-------------------|-----------------|------------------------|
| Full (Default) | `http://localhost:8000/v1` | `http://localhost:8000/openai/v1` | OAuth (Claude Code), API Key |
| Full (Explicit) | `http://localhost:8000/full/v1` | `http://localhost:8000/full/openai/v1` | OAuth (Claude Code), API Key |
| Minimal | `http://localhost:8000/min/v1` | `http://localhost:8000/min/openai/v1` | API Key only |
| Passthrough | `http://localhost:8000/pt/v1` | `http://localhost:8000/pt/openai/v1` | API Key only |

### Choosing the Right Mode

- **Full Mode**:
  - Required for OAuth authentication (Claude subscription users)
  - Works with Anthropic API keys
  - Injects Claude Code system prompt
  - Full header transformations

- **Minimal Mode**:
  - Does not work with OAuth (Claude subscription)
  - For Anthropic API key users only
  - No system prompt injection
  - Minimal header set

- **Passthrough Mode**:
  - Does not work with OAuth (Claude subscription)
  - For Anthropic API key users only
  - Minimal proxy interference
  - Direct API access

**Important**: OAuth credentials from Claude Code will return an error if used with `/min` or `/pt` modes.

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

### OAuth with Wrong Mode Error

When using OAuth credentials with `/min` or `/pt` modes:

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "system: This credential is only authorized for use with Claude Code and cannot be used for other API requests."
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
# OAuth Users (Claude Subscription) - MUST use full mode
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",  # Full mode (default)
    api_key="dummy"  # Ignored with OAuth
)

from anthropic import Anthropic
client = Anthropic(
    base_url="http://localhost:8000",  # Full mode (default)
    api_key="dummy"  # Ignored with OAuth
)

# API Key Users - Can use any mode
from anthropic import Anthropic

# Option 1: Full mode (with Claude Code features)
client = Anthropic(
    base_url="http://localhost:8000",  # or http://localhost:8000/full
    api_key="sk-ant-api03-..."
)

# Option 2: Minimal mode (lightweight, no Claude Code)
client = Anthropic(
    base_url="http://localhost:8000/min",
    api_key="sk-ant-api03-..."
)

# Option 3: Passthrough mode (direct API access)
client = Anthropic(
    base_url="http://localhost:8000/pt",
    api_key="sk-ant-api03-..."
)
```

### JavaScript/Node.js

```javascript
// OAuth Users (Claude Subscription) - MUST use full mode
import OpenAI from 'openai';
const client = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',  // Full mode (default)
  apiKey: 'dummy',  // Ignored with OAuth
});

// API Key Users - Can use any mode
import Anthropic from '@anthropic-ai/sdk';

// Option 1: Full mode (with Claude Code features)
const client = new Anthropic({
  baseURL: 'http://localhost:8000',  // or http://localhost:8000/full
  apiKey: 'sk-ant-api03-...',
});

// Option 2: Minimal mode (lightweight)
const client = new Anthropic({
  baseURL: 'http://localhost:8000/min',
  apiKey: 'sk-ant-api03-...',
});

// Option 3: Passthrough mode
const client = new Anthropic({
  baseURL: 'http://localhost:8000/pt',
  apiKey: 'sk-ant-api03-...',
});
```

### curl

```bash
# OAuth Users (Claude Subscription) - MUST use full mode
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'

# API Key Users - Full mode
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'

# API Key Users - Minimal mode
curl -X POST http://localhost:8000/min/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'

# OpenAI format with modes
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ant-api03-..." \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100}'
```

## Next Steps

- [Proxy Modes](../user-guide/proxy-modes.md) - Detailed guide on proxy transformation modes
- [Anthropic Endpoints](anthropic.md) - Detailed Anthropic API documentation
- [OpenAI Endpoints](openai.md) - Detailed OpenAI API documentation  
- [Models](models.md) - Available models and their capabilities
- [Health Check](health.md) - Health monitoring endpoints
