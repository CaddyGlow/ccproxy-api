# curl Examples

## Overview

Examples of using the Claude Code Proxy API with curl commands.

## OAuth Users (Claude Subscription)

### Basic Message (Anthropic Format - Full Mode)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

### Chat Completion (OpenAI Format - Full Mode)

```bash
curl -X POST http://localhost:8000/openai/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## API Key Users

### Full Mode (With Claude Code Features)

```bash
curl -X POST http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

### Minimal Mode (Lightweight, No Claude Code)

```bash
curl -X POST http://localhost:8000/min/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

### Passthrough Mode (Direct API Access)

```bash
curl -X POST http://localhost:8000/pt/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk-ant-api03-..." \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1000,
    "messages": [
      {"role": "user", "content": "Hello, Claude!"}
    ]
  }'
```

## Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

## With System Message

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a helpful assistant.",
    "messages": [
      {"role": "user", "content": "What is the capital of France?"}
    ]
  }'
```

## Multi-turn Conversation

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello!"},
      {"role": "assistant", "content": "Hello! How can I help you today?"},
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

## Check Available Models

```bash
curl -X GET http://localhost:8000/v1/models \
  -H "Content-Type: application/json"
```

## Health Check

```bash
curl -X GET http://localhost:8000/health
```

## Pretty Print JSON Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-20241022",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }' | jq .
```
