# Claude API Plugin

Connects CCProxy to Anthropic's Claude HTTP API with detection, health, and metrics.

## Highlights
- Wraps `ClaudeAPIAdapter` for chat, tool, and streaming requests
- Uses the detection service to discover CLI headers and available models
- Emits streaming metrics and standardized health checks via the hook registry

## Configuration
- `ClaudeAPISettings` defines base URLs, model cards, and auth manager name
- Works with `oauth_claude` or the credential balancer for token management
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin claude_api --config-class ClaudeAPISettings`

```toml
[plugins.claude_api]
# enabled = true
# base_url = "https://api.anthropic.com"
# auth_type = "oauth"
# supports_streaming = true
# include_sdk_content_as_xml = false
# system_prompt_injection_mode = "minimal"
```

## Related Components
- `adapter.py`: HTTP client for Anthropic endpoints
- `detection_service.py`: CLI and capability discovery helpers
- `routes.py`: FastAPI router mounted under `/claude/api`
