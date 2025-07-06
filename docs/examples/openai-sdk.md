# OpenAI SDK Examples

## Overview

Examples of using the Claude Code Proxy API with the OpenAI SDK, making it a drop-in replacement for OpenAI's API.

## Python OpenAI SDK

```python
from openai import OpenAI

# Configure client to use Claude proxy
client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"  # Proxy doesn't require OpenAI API key
)

# Chat completion
response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello, Claude!"}
    ]
)
print(response.choices[0].message.content)
```

## Node.js OpenAI SDK

```javascript
import OpenAI from 'openai';

// Configure client to use Claude proxy
const client = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'not-needed' // Proxy doesn't require OpenAI API key
});

// Chat completion
async function chat() {
  const response = await client.chat.completions.create({
    model: 'claude-3-5-sonnet-20241022',
    messages: [
      { role: 'user', content: 'Hello, Claude!' }
    ]
  });

  console.log(response.choices[0].message.content);
}

chat();
```

## Streaming with OpenAI SDK

### Python
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"
)

# Streaming chat completion
stream = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Tell me a story"}
    ],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
```

### Node.js
```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:8000/openai/v1',
  apiKey: 'not-needed'
});

async function streamChat() {
  const stream = await client.chat.completions.create({
    model: 'claude-3-5-sonnet-20241022',
    messages: [
      { role: 'user', content: 'Tell me a story' }
    ],
    stream: true
  });

  for await (const chunk of stream) {
    process.stdout.write(chunk.choices[0]?.delta?.content || '');
  }
}

streamChat();
```

## With Authentication

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="your-bearer-token"  # Use your configured bearer token
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
print(response.choices[0].message.content)
```

## Error Handling

```python
from openai import OpenAI, APIError

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"
)

try:
    response = client.chat.completions.create(
        model="claude-3-5-sonnet-20241022",
        messages=[
            {"role": "user", "content": "Hello!"}
        ]
    )
    print(response.choices[0].message.content)
except APIError as e:
    print(f"API Error: {e.message}")
    print(f"Status Code: {e.status_code}")
```

## System Messages

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
)
print(response.choices[0].message.content)
```

## List Models

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"
)

models = client.models.list()
for model in models.data:
    print(model.id)
```

## Temperature and Other Parameters

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/openai/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="claude-3-5-sonnet-20241022",
    messages=[
        {"role": "user", "content": "Write a creative story"}
    ],
    temperature=0.8,
    max_tokens=500
)
print(response.choices[0].message.content)
```
