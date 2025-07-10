#!/bin/bash

# OpenAI Format with Claude Code Options Example
#
# This example demonstrates using the standard OpenAI /v1/chat/completions endpoint
# with Claude Code specific options like allowed_tools, permission_mode, and cwd.

# Test OpenAI endpoint with Claude Code specific options
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {
        "role": "user",
        "content": "What is the current working directory? Please also tell me what files are in it."
      }
    ],
    "max_tokens": 2000,
    "max_thinking_tokens": 1000,
    "allowed_tools": ["Write", "Read", "LS"],
    "permission_mode": "acceptEdits",
    "append_system_prompt": "Focus on clean, well-documented code.",
    "cwd": "/tmp/claude-test"
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
#       "text": "/tmp/claude-test\n\n[File listing will be shown]"
#     }
#   ],
#   "model": "claude-3-5-sonnet-20241022",
#   "stop_reason": "end_turn",
#   "usage": {...}
# }

echo ""
echo "Claude Code specific options supported:"
echo "- allowed_tools: List of tools Claude can use"
echo "- permission_mode: How Claude handles edit permissions"
echo "- append_system_prompt: Additional system prompt text"
echo "- cwd: Working directory context"
echo "- max_thinking_tokens: Tokens for reasoning (Claude Code feature)"
