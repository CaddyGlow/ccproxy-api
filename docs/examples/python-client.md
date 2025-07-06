# Python Client Examples

## Overview

Examples of using the Claude Code Proxy API with Python client libraries.

## Using the Anthropic Python SDK

```python
from anthropic import Anthropic

# Configure client to use your proxy
client = Anthropic(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Proxy doesn't require API key
)

# Simple chat completion
response = client.messages.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.content[0].text)
```

## Using the OpenAI Python SDK

```python
from openai import OpenAI

# Configure client to use your proxy
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"  # Proxy doesn't require API key
)

# Simple chat completion
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
