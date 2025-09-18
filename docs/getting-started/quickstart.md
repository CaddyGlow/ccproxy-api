# Quick Start Guide

Get up and running with CCProxy API on your local machine in minutes.

## Prerequisites

Before starting, ensure you have:

- **Python 3.11 or higher**
- **Claude subscription** (Max, Pro, or Team)
- **Git** for cloning the repository (if installing from source)
- **Claude Code SDK** (optional, for SDK mode): `npm install -g @anthropic-ai/claude-code`

## Installation

### Quick Install (Recommended)

```bash
# Install with uv
uv tool install ccproxy-api

# Or with pipx
pipx install ccproxy-api
```

### Development Install

```bash
# Clone and setup
git clone https://github.com/CaddyGlow/ccproxy-api.git
cd ccproxy-api
make setup  # Installs dependencies and dev environment
```

## Authentication Setup

CCProxy supports multiple provider plugins, each with its own authentication:

### For Claude SDK Plugin

Uses the Claude Code SDK authentication:

```bash
# Login to Claude CLI (opens browser)
claude /login

# Verify Claude CLI status
claude /status

# Optional: Setup long-lived token
claude setup-token
```

### For Claude API Plugin

Uses CCProxy's OAuth2 authentication:

```bash
# Login via OAuth2 (opens browser)
ccproxy auth login

# Check authentication status
ccproxy auth status

# View detailed credential info
ccproxy auth info
```

### For Codex Plugin

Uses OpenAI OAuth2 PKCE authentication:

```bash
# Login to OpenAI (opens browser)
ccproxy auth login-openai

# Check status
ccproxy auth status
```

## Starting the Server

```bash
# Start the server (default port 8000)
ccproxy serve

# With custom port
ccproxy serve --port 8080

# Development mode with auto-reload
ccproxy serve --reload

# With debug logging
ccproxy serve --log-level debug

# Enable or disable plugins at startup
ccproxy serve --enable-plugin metrics --disable-plugin docker

# With verbose API logging
LOGGING__VERBOSE_API=true ccproxy serve
```

The server will start at `http://127.0.0.1:8000`

## Testing the API

### Quick Test - Claude SDK Mode

```bash
# Test with curl (Anthropic format)
curl -X POST http://localhost:8000/claude/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Say hello!"}
    ]
  }'

# Test with curl (OpenAI format)
curl -X POST http://localhost:8000/claude/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Say hello!"}
    ]
  }'
```

### Quick Test - Claude API Mode

```bash
# Direct API access (full control)
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 100,
    "messages": [
      {"role": "user", "content": "Say hello!"}
    ]
  }'
```

### Using with Python

```python
# Using OpenAI client library
from openai import OpenAI

# For Claude SDK mode
client = OpenAI(
    api_key="sk-dummy",  # Any dummy key
    base_url="http://localhost:8000/claude/v1"
)

# For Claude API mode  
client = OpenAI(
    api_key="sk-dummy",
    base_url="http://localhost:8000/api/v1"
)

# Make a request
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## Available Endpoints

### Claude SDK Plugin (`/claude`)
- `POST /claude/v1/messages` - Anthropic messages API
- `POST /claude/v1/chat/completions` - OpenAI chat completions
- Session support: `/claude/{session_id}/v1/...`

### Claude API Plugin (`/api`)
- `POST /api/v1/messages` - Anthropic messages API
- `POST /api/v1/chat/completions` - OpenAI chat completions
- `GET /api/v1/models` - List available models

### Codex Plugin (`/api/codex`)
- `POST /api/codex/responses` - Codex response API
- `POST /api/codex/chat/completions` - OpenAI format
- `POST /api/codex/{session_id}/responses` - Session-based responses
- `POST /api/codex/{session_id}/chat/completions` - Session-based completions
- `POST /api/codex/v1/chat/completions` - Standard OpenAI endpoint
- `GET /api/codex/v1/models` - List available models

## Monitoring & Debugging

### Health Check
```bash
curl http://localhost:8000/health
```

### Metrics (Prometheus format)
```bash
curl http://localhost:8000/metrics
```

Note: `/metrics` is provided by the metrics plugin. It is enabled by default when plugins are enabled.

### Enable Debug Logging
```bash
# Verbose API request/response logging
LOGGING__VERBOSE_API=true \
LOGGING__REQUEST_LOG_DIR=/tmp/ccproxy/request \
ccproxy serve --log-level debug

# View last request
ls -la /tmp/ccproxy/request/
```

## Common Issues

### Authentication Errors
- **Claude SDK**: Run `claude /login` or `claude setup-token`
- **Claude API**: Run `ccproxy auth login`
- **Codex**: Run `ccproxy auth login-openai`
- Check status: `ccproxy auth status`

### Port Already in Use
```bash
# Use a different port
ccproxy serve --port 8080
```

### Claude Code SDK Not Found
```bash
# Install Claude Code SDK
npm install -g @anthropic-ai/claude-code

# Or use API mode instead (doesn't require SDK)
# Just use /api endpoints instead of /claude
```

## Next Steps

- [API Usage Guide](../user-guide/api-usage.md) - Detailed API documentation
- [Authentication Guide](../user-guide/authentication.md) - Managing credentials
- [Configuration](configuration.md) - Advanced configuration options
- [Examples](../examples.md) - Code examples in various languages
