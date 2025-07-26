#!/usr/bin/env bash

curl -X POST "$OPENAI_BASE_URL/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
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
