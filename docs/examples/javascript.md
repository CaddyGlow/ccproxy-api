# JavaScript Examples

## Overview

Examples of using the Claude Code Proxy API with JavaScript and Node.js.

## Using fetch (Browser/Node.js)

```javascript
// Basic chat completion
async function chatWithClaude(message) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet-20241022',
      messages: [
        { role: 'user', content: message }
      ]
    })
  });

  const data = await response.json();
  return data.content[0].text;
}

// Usage
chatWithClaude('Hello, Claude!').then(response => {
  console.log(response);
});
```

## With Authentication

```javascript
async function authenticatedChat(message, bearerToken) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${bearerToken}`
    },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet-20241022',
      messages: [
        { role: 'user', content: message }
      ]
    })
  });

  const data = await response.json();
  return data.content[0].text;
}
```

## Streaming Response

```javascript
async function streamChat(message) {
  const response = await fetch('http://localhost:8000/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet-20241022',
      messages: [
        { role: 'user', content: message }
      ],
      stream: true
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6);
        if (data === '[DONE]') return;

        try {
          const parsed = JSON.parse(data);
          if (parsed.type === 'content_block_delta') {
            process.stdout.write(parsed.delta.text);
          }
        } catch (e) {
          // Skip invalid JSON
        }
      }
    }
  }
}
```

## Using Axios

```javascript
const axios = require('axios');

async function chatWithAxios(message) {
  try {
    const response = await axios.post('http://localhost:8000/v1/chat/completions', {
      model: 'claude-3-5-sonnet-20241022',
      messages: [
        { role: 'user', content: message }
      ]
    });

    return response.data.content[0].text;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}
```

## OpenAI Format

```javascript
async function openAIFormatChat(message) {
  const response = await fetch('http://localhost:8000/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'claude-3-5-sonnet-20241022',
      messages: [
        { role: 'user', content: message }
      ]
    })
  });

  const data = await response.json();
  return data.choices[0].message.content;
}
```

## React Hook Example

```javascript
import { useState, useCallback } from 'react';

function useClaude() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const chat = useCallback(async (message) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'claude-3-5-sonnet-20241022',
          messages: [
            { role: 'user', content: message }
          ]
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      return data.content[0].text;
    } catch (err) {
      setError(err.message);
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  return { chat, loading, error };
}
```
