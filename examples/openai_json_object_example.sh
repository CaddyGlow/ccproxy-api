#!/bin/bash

# OpenAI JSON Object Response Format Example
#
# This example demonstrates using the Claude Code Proxy API with OpenAI format
# and simple JSON object response formatting.

# Test JSON object format with programming content
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Generate a JSON object with information about a Python class for managing a database connection. Include class name, methods, and properties."
      }
    ],
    "max_tokens": 200,
    "response_format": {
      "type": "json_object"
    },
    "allowed_tools": ["str_replace_editor"]
  }' \
  -s | jq .

# Expected Response:
# {
#   "id": "msg_...",
#   "type": "message",
#   "role": "assistant",
#   "content": [
#     {
#       "type": "text",
#       "text": "{\n  \"class_name\": \"DatabaseConnection\",\n  \"methods\": [...],\n  \"properties\": [...]\n}"
#     }
#   ],
#   "model": "claude-3-5-sonnet-20241022",
#   "stop_reason": "end_turn",
#   "usage": {...}
# }

echo ""
echo "The response_format: {\"type\": \"json_object\"} instructs Claude to return"
echo "valid JSON. The OpenAI adapter automatically adds JSON formatting instructions"
echo "to the system prompt."
