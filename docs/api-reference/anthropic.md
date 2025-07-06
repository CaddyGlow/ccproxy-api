# Anthropic API Endpoints

Complete reference for Anthropic-compatible endpoints in your local Claude Code Proxy.

## Base URL

```
http://localhost:8000/v1
```

All Anthropic-compatible endpoints are prefixed with `/v1/`.

## Authentication

No API key required - uses your existing Claude CLI authentication automatically.

## Chat Completions

### POST /v1/chat/completions

Create a chat completion using Anthropic format.

!!! warning "Claude SDK Limitations"
    This proxy uses the **Claude Code SDK** internally, which has different capabilities than the official Anthropic API. Many standard API parameters are **ignored or not supported** because they cannot be passed to the Claude SDK. See the [Compatibility Notes](#compatibility-notes) section below for details.

#### Request

```http
POST /v1/chat/completions
Content-Type: application/json
```

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ],
  "max_tokens": 1000,
  "temperature": 0.7,
  "stream": false
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Claude model to use |
| `messages` | array | Yes | Array of message objects |
| `max_tokens` | integer | Yes | Maximum tokens to generate |
| `temperature` | float | No | Sampling temperature (0.0-1.0) |
| `top_p` | float | No | Nucleus sampling parameter |
| `top_k` | integer | No | Top-k sampling parameter |
| `stream` | boolean | No | Enable streaming responses |
| `stop_sequences` | array | No | Stop sequences for completion |
| `system` | string | No | System prompt |

#### Extended ClaudeCodeOptions Parameters (Unofficial)

!!! note "Unofficial Parameters"
    The following parameters are **not part of the official Anthropic API**. They are Claude Code SDK-specific extensions that allow you to configure advanced Claude Code options through the API.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `max_thinking_tokens` | integer | No | Maximum thinking tokens for Claude Code |
| `allowed_tools` | array | No | List of allowed tools |
| `disallowed_tools` | array | No | List of disallowed tools |
| `append_system_prompt` | string | No | Additional system prompt to append |
| `mcp_tools` | array | No | MCP tools to enable |
| `mcp_servers` | object | No | MCP server configurations |
| `permission_mode` | string | No | Permission mode: `default`, `acceptEdits`, or `bypassPermissions` |
| `continue_conversation` | boolean | No | Continue previous conversation |
| `resume` | string | No | Resume conversation ID |
| `max_turns` | integer | No | Maximum conversation turns |
| `permission_prompt_tool_name` | string | No | Permission prompt tool name |
| `cwd` | string | No | Working directory path |

**Example with Extended Parameters:**

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [
    {
      "role": "user", 
      "content": "Help me write some code"
    }
  ],
  "max_tokens": 1000,
  "permission_mode": "acceptEdits",
  "allowed_tools": ["Read", "Write", "Bash"],
  "max_thinking_tokens": 5000,
  "cwd": "/path/to/project"
}
```

#### Response

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "claude-3-5-sonnet-20241022",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 20,
    "total_tokens": 32
  }
}
```

#### Streaming Response

When `stream: true`, responses are sent as Server-Sent Events:

```
data: {"type": "message_start", "message": {...}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hello"}}

data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "!"}}

data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}}

data: [DONE]
```

## Models

### GET /v1/models

List available Claude models for your subscription.

#### Request

```http
GET /v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "claude-3-5-sonnet-20241022",
      "object": "model",
      "created": 1677649963,
      "owned_by": "anthropic"
    },
    {
      "id": "claude-3-5-haiku-20241022",
      "object": "model",
      "created": 1677649963,
      "owned_by": "anthropic"
    }
  ]
}
```

## Error Responses

### Error Format

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Invalid request parameters",
    "details": {
      "field": "model",
      "issue": "Model not found"
    }
  }
}
```

### Common Error Types

| Error Type | Status Code | Description |
|------------|-------------|-------------|
| `invalid_request_error` | 400 | Request validation failed |
| `authentication_error` | 401 | Authentication failed |
| `permission_error` | 403 | Permission denied |
| `not_found_error` | 404 | Resource not found |
| `rate_limit_error` | 429 | Rate limit exceeded |
| `internal_server_error` | 500 | Internal server error |
| `service_unavailable_error` | 503 | Service unavailable |

## Usage Examples

### Python with requests

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 100
    }
)

print(response.json())
```

### Python with Anthropic SDK

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy-key"  # Required but not used
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.content[0].text)
```

### curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### Streaming with curl

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Tell me a story"}],
    "max_tokens": 500,
    "stream": true
  }'

## Compatibility Notes

!!! important "Claude SDK vs Anthropic API Differences"
    This proxy uses the **Claude Code SDK** internally, which provides a different interface than the official Anthropic API. Understanding these differences is crucial for proper usage.

### Supported Parameters

These parameters are **fully supported** and passed to the Claude SDK:

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `model` | ✅ Full | Passed to Claude SDK |
| `messages` | ✅ Full | Converted to Claude SDK format |
| `max_tokens` | ✅ Full | Passed to Claude SDK |
| `system` | ✅ Full | Passed as system prompt |
| `stream` | ✅ Full | Handled by proxy streaming logic |

### Partially Supported Parameters

These parameters are **accepted but ignored** due to Claude SDK limitations:

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `temperature` | ⚠️ Ignored | Claude SDK doesn't support temperature control |
| `top_p` | ⚠️ Ignored | Claude SDK doesn't support nucleus sampling |
| `top_k` | ⚠️ Ignored | Claude SDK doesn't support top-k sampling |
| `stop_sequences` | ⚠️ Ignored | Claude SDK doesn't support custom stop sequences |

### Tool Parameters

Tool-related parameters have **limited support**:

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `tools` | ⚠️ Limited | Claude SDK has its own tool system - see ClaudeCodeOptions |
| `tool_choice` | ⚠️ Limited | Use `allowed_tools`/`disallowed_tools` instead |

### Unsupported Features

These Anthropic API features are **not available** through the Claude SDK:

- **Custom sampling parameters** (temperature, top_p, top_k)
- **Custom stop sequences** 
- **Tool definitions** (use Claude's built-in tools instead)
- **Function calling** (use Claude's tool system)
- **Image analysis** (depends on Claude SDK support)
- **PDF processing** (depends on Claude SDK support)

### Alternative: ClaudeCodeOptions

Instead of standard API parameters, use **ClaudeCodeOptions** for advanced control:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Help me code"}],
  "max_tokens": 2000,
  
  // Use ClaudeCodeOptions instead of standard API parameters
  "max_thinking_tokens": 10000,
  "allowed_tools": ["Read", "Write", "Bash"],
  "permission_mode": "acceptEdits",
  "append_system_prompt": "You are a coding assistant.",
  "cwd": "/path/to/project"
}
```

### Migration Considerations

If migrating from the official Anthropic API:

1. **Remove unsupported parameters**: temperature, top_p, top_k, stop_sequences
2. **Replace tool definitions**: Use Claude's built-in tools via `allowed_tools`
3. **Add ClaudeCodeOptions**: Use Claude-specific options for advanced features
4. **Test thoroughly**: Behavior may differ due to Claude SDK implementation

### Why These Limitations Exist

The Claude Code Proxy uses the **Claude Code SDK** which is designed for:
- **Interactive coding sessions** rather than general API usage
- **Built-in tool ecosystem** rather than custom functions
- **Local development workflows** rather than production API scenarios

This provides powerful coding capabilities but with different parameters than the standard Anthropic API.
```