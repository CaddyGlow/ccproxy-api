# Examples

Example code demonstrating how to use Claude Code Proxy API is available in the `examples/` directory.

## Available Examples

### Python Examples

#### `anthropic_tools_demo.py`
Demonstrates using the Anthropic SDK with tool/function calling features through the proxy.

```bash
cd examples
python anthropic_tools_demo.py
```

#### `openai_tools_demo.py`
Shows how to use the OpenAI SDK with the proxy, including function calling capabilities.

```bash
cd examples
python openai_tools_demo.py
```

### Interactive Chat Application

#### `textual_chat_agent.py`
A full-featured terminal chat application built with Textual that demonstrates:
- Real-time streaming responses
- Chat history management
- Interactive terminal UI
- Both Claude Code and API mode support

```bash
cd examples
python textual_chat_agent.py
```

See `examples/README_chat_agent.md` for detailed documentation.

## Quick Start Examples

### Using Anthropic SDK

```python
from anthropic import Anthropic

# Claude Code mode (default)
client = Anthropic(
    base_url="http://localhost:8000/sdk",
    api_key="dummy"  # Ignored with OAuth
)

# API mode (direct proxy)
client = Anthropic(
    base_url="http://localhost:8000/api",
    api_key="dummy"  # Ignored with OAuth
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=100
)
```

### Using OpenAI SDK

```python
from openai import OpenAI

# Claude Code mode (default)
client = OpenAI(
    base_url="http://localhost:8000/sdk/v1",
    api_key="dummy"  # Ignored with OAuth
)

# API mode (direct proxy)
client = OpenAI(
    base_url="http://localhost:8000/api/v1",
    api_key="dummy"  # Ignored with OAuth
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[{"role": "user", "content": "Hello!"}]
)
```

### Using curl

```bash
# Claude Code mode (default)
curl -X POST http://localhost:8000/sdk/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'

# API mode (direct proxy)
curl -X POST http://localhost:8000/api/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## More Examples

For more detailed examples and use cases, explore the `examples/` directory in the repository.
