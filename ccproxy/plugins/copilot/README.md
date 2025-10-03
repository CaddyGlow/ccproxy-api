# Copilot Plugin

Adds GitHub Copilot as a provider with OAuth, detection, and streaming support.

## Highlights
- Wraps the Copilot HTTP adapter while emitting OpenAI-compatible streams
- Manages OAuth exchange and token refresh through `CopilotOAuthProvider`
- Exposes GitHub-flavored routes under `/copilot` alongside v1 proxy APIs

## Configuration
- `CopilotConfig` controls base URLs, scopes, cache paths, and CLI detection
- Depends on `CopilotTokenManager` for credential storage and refresh logic
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin copilot --config-class CopilotConfig`

```toml
[plugins.copilot]
# enabled = true
# base_url = "https://api.githubcopilot.com"
# auth_type = "oauth"
# supports_streaming = true
# default_max_tokens = 4096
# account_type = "individual"
# request_timeout = 30
# max_retries = 3
# retry_delay = 1.0

[plugins.copilot.oauth]
# client_id = "Iv1.b507a08c87ecfe98"
# authorize_url = "https://github.com/login/device/code"
# token_url = "https://github.com/login/oauth/access_token"
# callback_port = 8080
# scopes = ["read:user"]
```

## Related Components
- `adapter.py`: request translation and HTTP execution layer
- `oauth/provider.py`: OAuth flow implementation for GitHub accounts
- `routes.py`: FastAPI routers for GitHub and proxy endpoints
