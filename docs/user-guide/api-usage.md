# API Usage

## Overview

The Claude Code Proxy API provides both Anthropic and OpenAI-compatible interfaces for Claude AI models.

## Anthropic API Format

### Base URL
```
http://localhost:8000/v1/
```

### Chat Completions
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## OpenAI API Format

### Base URL
```
http://localhost:8000/openai/v1/
```

### Chat Completions
```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
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

## Function Calling

The Claude Code Proxy API does not directly support function calling or tool use through the API endpoints. However, you can extend Claude's capabilities using MCP (Model Context Protocol) servers.

For detailed information on setting up and using MCP servers with Claude Code, see the [MCP Server Integration guide](mcp-integration.md).