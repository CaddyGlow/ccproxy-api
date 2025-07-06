# Streaming

## Overview

The Claude Code Proxy API supports streaming responses for real-time chat completions.

## Enabling Streaming

Set the `stream` parameter to `true` in your request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

## Response Format

Streaming responses are sent as Server-Sent Events (SSE):

```
data: {"type":"content_block_start","content_block":{"type":"text","text":""}}

data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Once"}}

data: {"type":"content_block_delta","delta":{"type":"text_delta","text":" upon"}}

data: [DONE]
```

## OpenAI Format Streaming

The OpenAI-compatible endpoint also supports streaming:

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

## Client Libraries

Most HTTP clients support SSE streaming. Examples:

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(line.decode())
```
