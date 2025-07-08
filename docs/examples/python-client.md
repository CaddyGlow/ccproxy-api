# Python Client Examples

## Overview

Examples of using the Claude Code Proxy API with Python client libraries.

## Using the Anthropic Python SDK

### OAuth Users (Claude Subscription)

```python
from anthropic import Anthropic

# OAuth users must use full mode (default)
client = Anthropic(
    base_url="http://localhost:8000",  # or http://localhost:8000/full
    api_key="dummy"  # Ignored with OAuth
)

# Simple message
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.content[0].text)
```

### API Key Users

```python
from anthropic import Anthropic

# Option 1: Full mode (with Claude Code features)
client = Anthropic(
    base_url="http://localhost:8000",
    api_key="sk-ant-api03-..."  # Your Anthropic API key
)

# Option 2: Minimal mode (lightweight, no Claude Code)
client = Anthropic(
    base_url="http://localhost:8000/min",
    api_key="sk-ant-api03-..."  # Your Anthropic API key
)

# Option 3: Passthrough mode (direct API access)
client = Anthropic(
    base_url="http://localhost:8000/pt",
    api_key="sk-ant-api03-..."  # Your Anthropic API key
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.content[0].text)
```

## Using the OpenAI Python SDK

### OAuth Users (Claude Subscription)

```python
from openai import OpenAI

# OAuth users must use full mode (default)
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",  # or /full/openai/v1
    api_key="dummy"  # Ignored with OAuth
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.choices[0].message.content)
```

### API Key Users

```python
from openai import OpenAI

# Option 1: Full mode
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="sk-ant-api03-..."  # Your Anthropic API key
)

# Option 2: Minimal mode
client = OpenAI(
    base_url="http://localhost:8000/min/openai/v1",
    api_key="sk-ant-api03-..."  # Your Anthropic API key
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.choices[0].message.content)
```

## Streaming Example

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Streaming response
stream = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.type == "content_block_delta":
        print(chunk.delta.text, end="")
```

## With Authentication

```python
from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="your-bearer-token"  # Use your configured bearer token
)

response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.content[0].text)
```

## Error Handling

```python
from anthropic import Anthropic, APIError

client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

try:
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.content[0].text)
except APIError as e:
    print(f"API Error: {e.message}")
    print(f"Status Code: {e.status_code}")
```
