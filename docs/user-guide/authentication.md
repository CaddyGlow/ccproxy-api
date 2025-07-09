# Authentication

## Overview

Claude Code Proxy API handles authentication in two layers:
1. **Claude Authentication**: Your Claude subscription credentials for accessing Claude AI
2. **API Authentication**: Optional token authentication for securing access to the proxy endpoints

## Important: Authentication Methods

CCProxy supports two different authentication methods that use separate credential storage:

### Claude CLI Authentication (Claude Code Mode)
- **Used by**: `claude /login` and `claude /status` commands
- **Storage locations**:
  - `~/.claude/credentials.json`
  - `~/.config/claude/credentials.json`
- **Purpose**: Authenticates the Claude CLI for Claude Code mode operations
- **Note**: These credentials are managed by the Claude CLI directly

### CCProxy Authentication (API Mode)
- **Used by**: `ccproxy auth` commands (login, validate, info)
- **Storage**:
  - **Primary**: System keyring (secure, recommended)
  - **Fallback**: `~/.config/ccproxy/credentials.json`
- **Purpose**: Authenticates for API mode operations using Anthropic OAuth2
- **Note**: Separate from Claude CLI credentials to avoid conflicts

## CCProxy Authentication Commands

Manage your CCProxy OAuth2 credentials with these commands:

### Login
```bash
ccproxy auth login
```
Opens a browser window for Anthropic OAuth2 authentication. Required for API mode access.

### Validate Credentials
```bash
ccproxy auth validate
```
Checks if your credentials are valid and shows:
- Subscription status and type
- Token expiration time
- Available OAuth scopes

### View Credential Info
```bash
ccproxy auth info
```
Displays detailed credential information and automatically renews the token if expired. Shows:
- Account email and organization
- Storage location (keyring or file)
- Token expiration and time remaining
- Access token (partially masked)

### Credential Storage
CCProxy credentials are stored securely:
- **Primary storage**: System keyring (when available)
- **Fallback storage**: `~/.config/ccproxy/credentials.json`
- Tokens are automatically managed and renewed by CCProxy

## API Authentication (Optional)

The proxy supports optional token authentication for securing access to the API endpoints. The proxy is designed to work seamlessly with the standard Anthropic and OpenAI client libraries without requiring any modifications.

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
