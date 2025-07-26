#!/usr/bin/env bash

curl -X POST "$ANTHROPIC_BASE_URL/v1/messages" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "anthropic-version: 2023-06-01" \
  --data '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
    {
      "role": "user",
      "content": "count to 5 in bash"
    }
  ],
  "stream": true
}' \
  --no-buffer
