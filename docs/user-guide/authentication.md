# Authentication

## Overview

The Claude Code Proxy API supports optional token authentication for securing access to the API endpoints. The proxy accepts multiple authentication header formats for compatibility with different clients.

## Supported Authentication Headers

The proxy accepts authentication tokens in these formats:
- **Anthropic Format**: `x-api-key: <token>` (takes precedence)
- **OpenAI/Bearer Format**: `Authorization: Bearer <token>`

All formats use the same configured `AUTH_TOKEN` value.

## Configuration

Set the `AUTH_TOKEN` environment variable:

```bash
export AUTH_TOKEN="your-secret-token-here"
```

Or add to your `.env` file:

```bash
echo "AUTH_TOKEN=your-secret-token-here" >> .env
```

## Usage Examples

### Anthropic Format (x-api-key)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-token-here" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ],
    "max_tokens": 100
  }'
```

### OpenAI/Bearer Format

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{
    "model": "claude-sonnet-4-20250514",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## Client SDK Examples

### Python with Anthropic Client

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy-key"  # Required but not used
)

# Using x-api-key header (recommended for Anthropic client)
client.default_headers = {"x-api-key": "your-secret-token-here"}

# Alternative: Using Bearer token
# client.default_headers = {"Authorization": "Bearer your-secret-token-here"}
```

### Python with OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"  # Required but not used
)

# Using Bearer token (recommended for OpenAI client)
client.default_headers = {"Authorization": "Bearer your-secret-token-here"}

# Alternative: Using x-api-key header
# client.default_headers = {"x-api-key": "your-secret-token-here"}
```

## No Authentication

If no `AUTH_TOKEN` is set, the API will accept all requests without authentication.

## Security Considerations

- Always use HTTPS in production
- Keep your bearer token secret and secure
- Consider using environment variables or secure secret management systems
- Rotate tokens regularly