curl -X POST "http://127.0.0.1:8000/api/codex/v1/chat/completions" -H "Content-Type: application/json" -v -d '{"model":"gpt-5","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024,"stream":true}'
curl -X POST "http://127.0.0.1:8000/api/codex/responses" -H "Content-Type: application/json" -v -d '{ "input": [ { "type": "message", "id": null, "role": "user", "content": [ { "type": "input_text", "text": "Hello" } ] } ], "model": "gpt-5", "stream": true, "store": false}'
curl -X POST "http://127.0.0.1:8000/claude/v1/chat/completions" -H "Content-Type: application/json" -v -d '{"model":"gpt-4","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024,"stream":true}'
curl -X POST "http://127.0.0.1:8000/claude/v1/messages" -H "Content-Type: application/json" -v -d '{"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100, "stream":true}'
curl -X POST "http://127.0.0.1:8000/api/v1/chat/completions" -H "Content-Type: application/json" -v -d '{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024,"stream":true}'
curl -X POST "http://127.0.0.1:8000/api/v1/messages" -H "Content-Type: application/json" -v -d '{"model": "claude-sonnet-4-20250514", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 100, "stream":true}'
p
