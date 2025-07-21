#!/bin/bash

echo "Testing Claude Code Proxy API Streaming with curl"
echo "================================================="

# Test streaming endpoint
curl -N -v -X POST http://127.0.0.1:8000/sdk/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -H "anthropic-version: 2023-06-01" \
  -H "x-api-key: dummy" \
  -d '{
    "model": "gpt-4o",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": "Write a short story about a robot learning to paint. Make it creative and engaging."
      }
    ],
    "stream": true
  }'
