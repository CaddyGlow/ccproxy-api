# Authentication

## Overview

CCProxy API handles authentication in multiple layers:
1. **Claude Authentication**: Your Claude subscription credentials for accessing Claude AI
2. **OpenAI/Codex Authentication**: Your ChatGPT Plus credentials for accessing the Response API
3. **API Authentication**: Optional token authentication for securing access to the proxy endpoints

## Important: Authentication Methods

CCProxy supports multiple authentication methods with separate credential storage:

### Claude CLI Authentication (Claude Code Mode)
- **Used by**: `claude /login` and `claude /status` commands
- **Storage locations**:
  - `~/.claude/credentials.json`
  - `~/.config/claude/credentials.json`
- **Purpose**: Authenticates the Claude CLI for Claude Code mode operations
- **Note**: These credentials are managed by the Claude CLI directly

### CCProxy Claude Authentication (API Mode)
- **Used by**: `ccproxy auth` commands (login, validate, info)
- **Storage**: `~/.config/ccproxy/credentials.json`
- **Purpose**: Authenticates for API mode operations using Anthropic OAuth2
- **Note**: Separate from Claude CLI credentials to avoid conflicts

### OpenAI/Codex Authentication (Response API)
- **Used by**: `ccproxy auth login-openai` and Codex Response API endpoints
- **Storage**: `$HOME/.codex/auth.json`
- **Purpose**: Authenticates for ChatGPT Plus Response API access
- **Requirements**: Active ChatGPT Plus subscription
- **Credential Reuse**: Automatically uses existing Codex CLI credentials if available

## Authentication Commands

### Claude Authentication Commands

Manage your Claude OAuth2 credentials:

#### Login
```bash
ccproxy auth login
```
Opens a browser window for Anthropic OAuth2 authentication. Required for API mode access.

#### Validate Credentials
```bash
ccproxy auth validate
```
Checks if your credentials are valid and shows:
- Subscription status and type
- Token expiration time
- Available OAuth scopes

#### View Credential Info
```bash
ccproxy auth info
```
Displays detailed credential information and automatically renews the token if expired. Shows:
- Account email and organization
- Storage location (file)
- Token expiration and time remaining
- Access token (partially masked)

### OpenAI/Codex Authentication Commands

Manage your ChatGPT Plus credentials:

#### Login
```bash
# Enable Codex provider first
ccproxy config codex --enable

# Login with OpenAI OAuth2 (opens browser)
ccproxy auth login-openai
```

Uses OAuth2 PKCE flow to authenticate with OpenAI. The process:
1. Opens browser for authentication
2. Requires ChatGPT Plus subscription
3. Stores credentials in `$HOME/.codex/auth.json`
4. Automatically refreshes tokens when needed

#### Alternative: Use Codex CLI
```bash
# Install Codex CLI if not present
npm install -g @openai/codex-cli

# Authenticate with Codex
codex auth login

# CCProxy will automatically detect and use these credentials
```

#### Check All Authentication Status
```bash
ccproxy auth status
```
Shows authentication status for all providers:
- Claude SDK credentials
- Claude API credentials  
- OpenAI/Codex credentials

#### View Detailed OpenAI Credentials
```bash
ccproxy auth openai-info
```

Example output showing valid ChatGPT Plus credentials:
```
                OpenAI Credential Information

     OpenAI Account
       L Account ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
       L Status: Active
       L Plan Type: PLUS
       L User ID: user-xxxxxxxxxxxxxxxxxxxx

                            Token Details
     ╭──────────────────┬──────────────────────────────────────╮
     │ Property         │ Value                                │
     ├──────────────────┼──────────────────────────────────────┤
     │ Storage Location │ /home/user/.codex/auth.json         │
     │ Algorithm        │ RS256                                │
     │ Token Type       │ JWT                                  │
     │ Key ID           │ xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx │
     │ Token Expired    │ No                                   │
     │ Expires At       │ 2025-08-22 11:53:34 UTC              │
     │ Time Remaining   │ 9 days, 3 hours, 26 minutes          │
     │ Issued At        │ 2025-08-12 11:53:33 UTC              │
     │ Issuer           │ https://auth.openai.com              │
     │ Audience         │ https://api.openai.com/v1            │
     │ JWT ID           │ xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx │
     │ Access Token     │ eyJhbGciOiJS...xIXcUrNE              │
     │ Refresh Token    │ Available                            │
     ╰──────────────────┴──────────────────────────────────────╯
```

This confirms:
- **PLUS Plan**: ChatGPT Plus subscription is active
- **Token Valid**: Access token is not expired
- **Auto-Renewal**: Refresh token is available for automatic renewal
- **Storage**: Credentials are stored in `$HOME/.codex/auth.json`

### Credential Storage Locations

#### Claude Credentials
- **Storage**: `~/.config/ccproxy/credentials.json`
- Tokens are automatically managed and renewed by CCProxy

#### OpenAI/Codex Credentials
- **Storage**: `$HOME/.codex/auth.json`
- **Format**: JSON with access_token, refresh_token, expires_at
- **Sharing**: Credentials shared with official Codex CLI
- **Auto-refresh**: Tokens renewed automatically when expired

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

All formats use the same configured `SECURITY__AUTH_TOKEN` value.

## Configuration

Set the `SECURITY__AUTH_TOKEN` environment variable:

```bash
export SECURITY__AUTH_TOKEN="your-secret-token-here"
```

Or add to your `.env` file:

```bash
echo "SECURITY__AUTH_TOKEN=your-secret-token-here" >> .env
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

If no `SECURITY__AUTH_TOKEN` is set, the API will accept all requests without authentication.

## Security Considerations

- Always use HTTPS in production
- Keep your bearer token secret and secure
- Consider using environment variables or secure secret management systems
- Rotate tokens regularly
