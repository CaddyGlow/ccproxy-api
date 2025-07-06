# OpenAI API Endpoints

Complete reference for OpenAI-compatible endpoints in your local Claude Code Proxy.

## Base URL

```
http://localhost:8000/openai/v1
```

All OpenAI-compatible endpoints are prefixed with `/openai/v1/`.

## Authentication

No API key required - uses your existing Claude CLI authentication automatically.

**Note**: OpenAI client libraries require an API key parameter, but any dummy value will work (e.g., "dummy-key").

## Chat Completions

### POST /openai/v1/chat/completions

Create a chat completion using OpenAI format, automatically translated to Claude.

!!! warning "Claude SDK Limitations"
    This proxy uses the **Claude Code SDK** internally, not the official OpenAI or Anthropic APIs. Many OpenAI parameters are **ignored or not supported** because they cannot be passed to the Claude SDK. See the [Claude SDK Compatibility](#claude-sdk-compatibility) section below for details.

#### Request

```http
POST /openai/v1/chat/completions
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
| `max_tokens` | integer | No | Maximum tokens to generate |
| `temperature` | float | No | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | Nucleus sampling parameter |
| `stream` | boolean | No | Enable streaming responses |
| `stop` | array/string | No | Stop sequences for completion |
| `presence_penalty` | float | No | Presence penalty (-2.0 to 2.0) |
| `frequency_penalty` | float | No | Frequency penalty (-2.0 to 2.0) |
| `user` | string | No | User identifier |

#### Extended ClaudeCodeOptions Parameters (Unofficial)

!!! note "Unofficial Parameters"
    The following parameters are **not part of the official OpenAI API**. They are Claude Code SDK-specific extensions that allow you to configure advanced Claude Code options through the API.

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

When `stream: true`, responses are sent as Server-Sent Events in OpenAI format:

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{"content":"!"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"claude-3-5-sonnet-20241022","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

## Models

### GET /openai/v1/models

List available Claude models in OpenAI format.

#### Request

```http
GET /openai/v1/models
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

## Parameter Translation

The proxy automatically translates OpenAI parameters to Claude equivalents:

| OpenAI Parameter | Claude Parameter | Notes |
|------------------|------------------|-------|
| `max_tokens` | `max_tokens` | Direct mapping |
| `temperature` | `temperature` | Clamped to 0.0-1.0 range |
| `top_p` | `top_p` | Direct mapping |
| `stop` | `stop_sequences` | Converted to array format |
| `presence_penalty` | Not supported | Ignored with warning |
| `frequency_penalty` | Not supported | Ignored with warning |
| `user` | Not supported | Ignored |

## Error Responses

### Error Format

```json
{
  "error": {
    "message": "Invalid request parameters",
    "type": "invalid_request_error",
    "param": "model",
    "code": "invalid_model"
  }
}
```

### Common Error Codes

| Error Code | Status Code | Description |
|------------|-------------|-------------|
| `invalid_request_error` | 400 | Request validation failed |
| `invalid_api_key` | 401 | Authentication failed |
| `insufficient_quota` | 403 | Permission denied |
| `model_not_found` | 404 | Model not available |
| `rate_limit_exceeded` | 429 | Rate limit exceeded |
| `internal_server_error` | 500 | Internal server error |
| `service_unavailable` | 503 | Service unavailable |

## Usage Examples

### Python with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"  # Required but not used
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)

print(response.choices[0].message.content)
```

### Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy-key"
)

stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True,
    max_tokens=500
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### JavaScript/Node.js

```javascript
import OpenAI from 'openai';

const openai = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'dummy-key' // Required but not used
});

const completion = await openai.chat.completions.create({
  model: 'claude-3-5-sonnet-20241022',
  messages: [{ role: 'user', content: 'Hello!' }],
  max_tokens: 100
});

console.log(completion.choices[0].message.content);
```

### curl

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-key" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Migration from OpenAI

To migrate existing OpenAI-based applications:

1. **Change the base URL**:
   ```python
   # Before
   client = OpenAI(api_key="your-openai-key")

   # After
   client = OpenAI(
       base_url="http://localhost:8000/openai/v1",
       api_key="dummy-key"
   )
   ```

2. **Update model names** to Claude models:
   ```python
   # Before
   model="gpt-4-turbo"

   # After
   model="claude-3-5-sonnet-20241022"
   ```

3. **Remove unsupported parameters** (optional):
   - `presence_penalty`
   - `frequency_penalty`
   - `user`

That's it! Your existing OpenAI code should work without other changes.

## Claude SDK Compatibility

!!! important "Understanding Claude SDK Limitations"
    This proxy uses the **Claude Code SDK** internally, which provides different capabilities than both OpenAI and Anthropic APIs. Many standard parameters are **ignored or not supported**.

### Fully Supported Parameters

These OpenAI parameters work correctly:

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `model` | ✅ Full | Mapped to Claude models |
| `messages` | ✅ Full | Converted to Claude SDK format |
| `max_tokens` | ✅ Full | Passed to Claude SDK |
| `stream` | ✅ Full | Handled by proxy streaming logic |

### Ignored Parameters

These parameters are **accepted but completely ignored** due to Claude SDK limitations:

| Parameter | Support Level | Reason |
|-----------|---------------|--------|
| `temperature` | ❌ Ignored | Claude SDK doesn't support temperature control |
| `top_p` | ❌ Ignored | Claude SDK doesn't support nucleus sampling |
| `presence_penalty` | ❌ Ignored | Claude SDK doesn't support penalty parameters |
| `frequency_penalty` | ❌ Ignored | Claude SDK doesn't support penalty parameters |
| `logit_bias` | ❌ Ignored | Claude SDK doesn't support logit manipulation |
| `stop` | ❌ Ignored | Claude SDK doesn't support custom stop sequences |
| `user` | ❌ Ignored | Claude SDK doesn't track user identifiers |
| `seed` | ❌ Ignored | Claude SDK doesn't support deterministic sampling |
| `logprobs` | ❌ Ignored | Claude SDK doesn't provide log probabilities |
| `top_logprobs` | ❌ Ignored | Claude SDK doesn't provide log probabilities |

### Limited Tool Support

OpenAI tool parameters have very limited support:

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `tools` | ⚠️ Very Limited | Claude SDK has its own tool ecosystem |
| `tool_choice` | ⚠️ Very Limited | Use ClaudeCodeOptions instead |
| `parallel_tool_calls` | ❌ Ignored | Claude SDK controls tool execution |

### Response Format Limitations

| Parameter | Support Level | Notes |
|-----------|---------------|-------|
| `response_format` | ❌ Ignored | Claude SDK doesn't support structured output |
| `n` | ❌ Ignored | Claude SDK only generates single responses |

### Using ClaudeCodeOptions Instead

For advanced control, use ClaudeCodeOptions parameters with OpenAI format:

```json
{
  "model": "claude-3-5-sonnet-20241022",
  "messages": [{"role": "user", "content": "Help me code"}],
  "max_tokens": 2000,

  // Claude-specific options work with OpenAI format
  "max_thinking_tokens": 10000,
  "allowed_tools": ["Read", "Write", "Bash"],
  "permission_mode": "acceptEdits",
  "cwd": "/path/to/project"
}
```

### Migration Impact

When migrating from OpenAI to this proxy:

**✅ Will Work:**
- Basic chat completions
- Message history
- Streaming responses
- Model selection (mapped to Claude models)

**❌ Will Be Ignored:**
- All sampling parameters (temperature, top_p, etc.)
- Penalty parameters
- Custom stop sequences
- Deterministic generation (seed)
- Log probabilities
- Multiple response generation

**⚠️ Limited Functionality:**
- Tool/function calling (use Claude's built-in tools)
- Structured output (response_format)

### Why These Limitations Exist

The Claude Code SDK is designed for:
- **Interactive development workflows**
- **Built-in coding tools** (Read, Write, Bash, etc.)
- **Local project contexts**
- **Permission-based tool access**

This is fundamentally different from OpenAI's API which focuses on:
- **Configurable text generation**
- **Custom function definitions**
- **Fine-grained sampling control**
- **Structured output formats**

### Recommendations

1. **For basic chat**: OpenAI format works fine, ignore unsupported parameters
2. **For coding tasks**: Use ClaudeCodeOptions for powerful development features
3. **For production**: Consider whether Claude SDK limitations meet your needs
4. **For migration**: Test thoroughly and remove unsupported parameter usage
