# API Usage

## Overview

The Claude Code Proxy API provides both Anthropic and OpenAI-compatible interfaces for Claude AI models with three transformation modes.

## Proxy Modes

| Mode | URL Prefix | Authentication | Use Case |
|------|------------|----------------|----------|
| Full | `/` or `/full/` | OAuth, API Key | Claude Code features, OAuth users |
| Minimal | `/min/` | API Key only | Lightweight proxy |
| Passthrough | `/pt/` | API Key only | Direct API access |

**Important**: OAuth credentials from Claude Code only work with full mode.

## Anthropic API Format

### Base URLs by Mode
```
Full Mode:    http://localhost:8000/v1/
Minimal Mode: http://localhost:8000/min/v1/
Passthrough:  http://localhost:8000/pt/v1/
```

### Messages Endpoint
```bash
# OAuth users (full mode only)
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'

# API key users (any mode)
curl -X POST http://localhost:8000/min/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## OpenAI API Format

### Base URLs by Mode
```
Full Mode:    http://localhost:8000/openai/v1/
Minimal Mode: http://localhost:8000/min/openai/v1/
Passthrough:  http://localhost:8000/pt/openai/v1/
```

### Chat Completions
```bash
# OAuth users (full mode only)
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'

# API key users (minimal mode)
curl -X POST http://localhost:8000/min/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## Supported Models

- claude-3-5-sonnet-20241022
- claude-3-5-haiku-20241022
- claude-3-opus-20240229
- claude-3-sonnet-20240229
- claude-3-haiku-20240307

## Function Calling

The Claude Code Proxy API does not directly support function calling or tool use through the API endpoints. However, you can extend Claude's capabilities using MCP (Model Context Protocol) servers.

For detailed information on setting up and using MCP servers with Claude Code, see the [MCP Server Integration guide](mcp-integration.md).
