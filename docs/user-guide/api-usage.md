# API Usage

## Overview

The Claude Code Proxy API is a reverse proxy to api.anthropic.com that provides both Anthropic and OpenAI-compatible interfaces. It offers two main access modes:

## Access Modes

| Mode | URL Prefix | Method | Use Case |
|------|------------|--------|----------|
| Claude Code | `/` or `/cc/` | Uses claude-code-sdk | Access to all Claude Code tools |
| API | `/api/` | Direct proxy with headers | Full API control and settings |

## Anthropic API Format

### Base URLs by Mode
```
Claude Code Mode: http://localhost:8000/v1/
API Mode:         http://localhost:8000/api/v1/
```

### Messages Endpoint
```bash
# Claude Code mode (default) - with all tools
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'

# API mode - direct proxy for full control
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
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
Claude Code Mode: http://localhost:8000/v1/
API Mode:         http://localhost:8000/api/v1/
```

### Chat Completions
```bash
# Claude Code mode (default) - with all tools
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'

# API mode - direct proxy for full control
curl -X POST http://localhost:8000/api/v1/chat/completions \
  -H "Content-Type: application/json" \
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

## Function Calling and Tools

### Claude Code Mode Limitations
When using Claude Code mode (default), the proxy uses the claude-code-sdk which has these limitations:
- Cannot directly use ToolCall/function calling through the API
- Limited control over model settings
- However, you get access to all tools configured in Claude Code

### API Mode Advantages
When using API mode (`/api/*`), you get:
- Full access to all API features including ToolCall
- Complete control over model settings
- Direct pass-through to api.anthropic.com

### MCP Integration
You can extend Claude's capabilities using MCP (Model Context Protocol) servers in Claude Code mode.

For detailed information on setting up and using MCP servers with Claude Code, see the [MCP Server Integration guide](mcp-integration.md).
