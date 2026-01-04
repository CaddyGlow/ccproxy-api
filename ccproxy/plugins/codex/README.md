# Codex Plugin

Integrates the Codex provider with CCProxy using OpenAI-compatible formats.

## Highlights
- Uses `CodexAdapter` for chat, responses, and streaming endpoints
- Supports OAuth token rotation via `CodexTokenManager`
- Registers a streaming metrics hook that can enrich data with pricing info

## Configuration
- `CodexSettings` manages base URL, model catalog, and streaming support
- Pair with `oauth_codex` or the credential balancer for credential management
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin codex --config-class CodexSettings`

```toml
[plugins.codex]
# enabled = true
# base_url = "https://chatgpt.com/backend-api/codex"
# auth_type = "oauth"
# supports_streaming = true
# preferred_upstream_mode = "streaming"
# supported_input_formats = ["openai.responses", "openai.chat_completions", "anthropic.messages"]
# verbose_logging = false

[plugins.codex.oauth]
# base_url = "https://auth.openai.com"
# client_id = "app_EMoamEEZ73f0CkXaXp7hrann"
# callback_port = 1455
```

## Related Components
- `detection_service.py`: detects CLI capabilities and default headers
- `hooks.py`: streaming metrics extraction logic
- `routes.py`: FastAPI router mounted under `/codex`
