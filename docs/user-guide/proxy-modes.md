# Proxy Transformation Modes

The Claude Code Proxy API supports three transformation modes to accommodate different use cases and authentication methods.

## Authentication Compatibility

| Mode | OAuth (Claude Subscription) | Anthropic API Key |
|------|----------------------------|-------------------|
| Full | Yes | Yes |
| Minimal | No | Yes |
| Passthrough | No | Yes |

## Full Mode (Default)

**URL Prefixes:** `/` (root) or `/full/`

Full mode provides complete transformations required for OAuth authentication:
- Injects Claude Code system prompt (required for OAuth)
- Adds all necessary Claude CLI headers
- Handles format conversion between OpenAI and Anthropic
- **Required for Claude subscription users**

**When to use:**
- Using OAuth authentication (Claude subscription) - **REQUIRED**
- Using Anthropic API keys with full features
- Need Claude Code functionality

**Example:**
```bash
# OAuth users (Claude subscription)
export OPENAI_BASE_URL="http://localhost:8000/openai/v1"

# API key users who want full features
export ANTHROPIC_BASE_URL="http://localhost:8000"
export ANTHROPIC_API_KEY="sk-ant-..."
```

## Minimal Mode

**URL Prefix:** `/min/`

**Important**: This mode does NOT work with OAuth authentication from Claude Code.

Minimal mode provides basic proxy functionality:
- Basic authentication headers
- No Claude Code system prompt (incompatible with OAuth)
- Preserves original request body
- **API key authentication only**

**When to use:**
- Using Anthropic API keys
- Need lightweight proxy without Claude Code features
- NOT for OAuth/Claude subscription users

**Example:**
```bash
# API key users only
export ANTHROPIC_BASE_URL="http://localhost:8000/min"
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Error with OAuth

If you try to use minimal mode with OAuth credentials:
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "system: This credential is only authorized for use with Claude Code and cannot be used for other API requests."
  }
}
```

## Passthrough Mode

**URL Prefix:** `/pt/`

**Important**: This mode does NOT work with OAuth authentication from Claude Code.

Passthrough mode provides minimal transformation:
- Direct API access with minimal proxy interference
- No body transformation
- **API key authentication only**

**When to use:**
- Using Anthropic API keys
- Need direct API access
- Debugging API calls
- NOT for OAuth/Claude subscription users

### Error with OAuth

Same as minimal mode - OAuth credentials will be rejected with an error about being restricted to Claude Code usage.

## Quick Reference

### For Claude Subscription Users (OAuth):
Always use full mode (root or `/full/`):
```python
# Python
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="dummy"  # Ignored with OAuth
)

from anthropic import Anthropic
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="dummy"  # Ignored with OAuth
)
```

### For API Key Users:
Can use any mode:
```python
# Full mode (with Claude Code features)
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="sk-ant-..."
)

# Minimal mode (lightweight)
client = Anthropic(
    base_url="http://localhost:8000/min",
    api_key="sk-ant-..."
)

# Passthrough mode (direct API)
client = Anthropic(
    base_url="http://localhost:8000/pt",
    api_key="sk-ant-..."
)
```

## Mode Selection Guide

| Your Setup | Required Mode | Example URL |
|------------|---------------|-------------|
| Claude Subscription (OAuth) | Full | `http://localhost:8000/v1/messages` |
| Anthropic API Key + Claude features | Full | `http://localhost:8000/v1/messages` |
| Anthropic API Key + Lightweight | Minimal | `http://localhost:8000/min/v1/messages` |
| Anthropic API Key + Direct API | Passthrough | `http://localhost:8000/pt/v1/messages` |

## Common Errors

### OAuth with Wrong Mode

**Problem:** Using `/min` or `/pt` endpoints with OAuth credentials

**Error:**
```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "system: This credential is only authorized for use with Claude Code and cannot be used for other API requests."
  }
}
```

**Solution:** Use full mode endpoints:
- `http://localhost:8000/v1/messages`
- `http://localhost:8000/full/v1/messages`

### Example Commands

**This fails with OAuth:**
```bash
curl -X POST http://localhost:8000/min/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'
```

**This works with OAuth:**
```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-3-5-sonnet-20241022", "messages": [{"role": "user", "content": "Hello"}]}'
```
