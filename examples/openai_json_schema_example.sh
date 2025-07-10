#!/bin/bash

# OpenAI JSON Schema Response Format Example
#
# This example demonstrates how to use the Claude Code Proxy API with OpenAI format
# and JSON schema response formatting. The API converts OpenAI requests to Anthropic
# format while maintaining Claude Code identity and behavior.

# Test JSON schema with programming-related content
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "Generate a JSON object describing a Python function that calculates fibonacci numbers."
      }
    ],
    "max_tokens": 300,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "function_info",
        "schema": {
          "type": "object",
          "properties": {
            "function_name": {"type": "string"},
            "parameters": {"type": "array", "items": {"type": "string"}},
            "return_type": {"type": "string"},
            "description": {"type": "string"},
            "complexity": {"type": "string"}
          },
          "required": ["function_name", "parameters", "return_type", "description"]
        }
      }
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
#       "text": "{\n  \"function_name\": \"calculate_fibonacci\",\n  \"parameters\": [...],\n  \"return_type\": \"int\",\n  \"description\": \"...\",\n  \"complexity\": \"...\"\n}"
#     }
#   ],
#   "model": "claude-3-5-sonnet-20241022",
#   "stop_reason": "end_turn",
#   "usage": {...}
# }

echo ""
echo "Note: The 'allowed_tools' field is required to route the request through"
echo "the SDK path which supports OAuth authentication. Without tools, requests"
echo "use the proxy path which requires API key authentication."
