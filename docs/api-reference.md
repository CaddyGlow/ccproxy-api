# API Reference

Claude Code Proxy provides multiple endpoint modes for different use cases.

## Endpoint Modes

### Claude Code Mode (Default)
- **Base URL**: `http://localhost:8000/` or `http://localhost:8000/cc/`
- **Method**: Uses the official claude-code-sdk
- **Limitations**: Cannot use ToolCall, limited model settings control
- **Advantages**: Access to all Claude Code tools and features

### API Mode (Direct Proxy)
- **Base URL**: `http://localhost:8000/api/`
- **Method**: Direct reverse proxy to api.anthropic.com
- **Features**: Full API access, all model settings available
- **Authentication**: Injects OAuth headers automatically

## Supported Endpoints

### Anthropic Format
```
POST /v1/messages         # Claude Code mode (default)
POST /api/v1/messages     # API mode (direct proxy)
POST /cc/v1/messages      # Claude Code mode (explicit)
```

### OpenAI Compatibility Layer
```
POST /openai/v1/chat/completions       # Claude Code mode (default)
POST /api/openai/v1/chat/completions   # API mode (direct proxy)
POST /cc/openai/v1/chat/completions    # Claude Code mode (explicit)
```

### Utility Endpoints
```
GET /health              # Health check
GET /v1/models           # List available models
GET /openai/v1/models    # List models (OpenAI format)
```

## Available Models

Models available depend on your Claude subscription:

- `claude-opus-4-20250514` - Claude 4 Opus (most capable)
- `claude-sonnet-4-20250514` - Claude 4 Sonnet (latest)
- `claude-3-7-sonnet-20250219` - Claude 3.7 Sonnet
- `claude-3-5-sonnet-20241022` - Claude 3.5 Sonnet
- `claude-3-5-sonnet-20240620` - Claude 3.5 Sonnet (legacy)

## Request Format

### Anthropic Format
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1000
}
```

### OpenAI Format
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 1000,
  "temperature": 0.7
}
```

## Authentication

- **OAuth Users**: No API key needed, uses Claude subscription
- **API Key Users**: Include `x-api-key` header or `Authorization: Bearer` header

## Streaming

Both modes support streaming responses:
```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

## Mode Selection Guide

| Use Case | Recommended Mode | Endpoint |
|----------|------------------|----------|
| Need Claude Code tools | Claude Code mode | `/v1/messages` |
| Need full API control | API mode | `/api/v1/messages` |
| Using OpenAI SDK | Either mode | `/openai/v1/chat/completions` |
| Direct API access | API mode | `/api/v1/messages` |
