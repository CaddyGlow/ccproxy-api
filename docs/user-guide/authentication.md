# Authentication

## Overview

The Claude Code Proxy API supports optional token authentication for securing access to the API endpoints. The proxy is designed to work seamlessly with the standard Anthropic and OpenAI client libraries without requiring any modifications.

## Why Multiple Authentication Formats?

Different AI client libraries use different authentication header formats:
- **Anthropic SDK**: Sends the API key as `x-api-key` header
- **OpenAI SDK**: Sends the API key as `Authorization: Bearer` header

By supporting both formats, you can:
1. **Use standard libraries as-is**: No need to modify headers or use custom configurations
2. **Secure your proxy**: Add authentication without breaking compatibility
3. **Switch between clients easily**: Same auth token works with any client library

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

# Just use the standard Anthropic client - no modifications needed!
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="your-secret-token-here"  # Automatically sent as x-api-key header
)

# Make requests normally
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Python with OpenAI Client

```python
from openai import OpenAI

# Just use the standard OpenAI client - no modifications needed!
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="your-secret-token-here"  # Automatically sent as Bearer token
)

# Make requests normally
response = client.chat.completions.create(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### JavaScript/TypeScript with OpenAI SDK

```javascript
import OpenAI from 'openai';

// Standard OpenAI client setup
const openai = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'your-secret-token-here',  // Automatically sent as Bearer token
});

// Use normally
const response = await openai.chat.completions.create({
  model: 'claude-sonnet-4-20250514',
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

## No Authentication

If no `AUTH_TOKEN` is set, the API will accept all requests without authentication.

## Security Considerations

- Always use HTTPS in production
- Keep your bearer token secret and secure
- Consider using environment variables or secure secret management systems
- Rotate tokens regularly
