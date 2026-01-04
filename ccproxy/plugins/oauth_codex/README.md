# OAuth Codex Plugin

OAuth-only plugin that issues and refreshes tokens for the Codex provider.

## Highlights
- Provides `CodexOAuthProvider` implementing the OAuth dance for Codex APIs
- Registers an auth manager consumable by `codex` and related plugins
- Leverages shared HTTP client and hook infrastructure from the container

## Configuration
- `CodexOAuthConfig` configures client ID, secrets, scopes, and cache paths
- Enable when using Codex endpoints that require OAuth tokens
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin oauth_codex --config-class CodexOAuthConfig`

```toml
[plugins.oauth_codex]
# enabled = true
# base_url = "https://auth.openai.com"
# token_url = "https://auth.openai.com/oauth/token"
# authorize_url = "https://auth.openai.com/oauth/authorize"
# profile_url = "https://api.openai.com/oauth/profile"
# client_id = "app_EMoamEEZ73f0CkXaXp7hrann"
# redirect_uri = "http://localhost:1455/auth/callback"
# scopes = ["openid", "profile", "email", "offline_access"]
# headers = {"User-Agent" = "Codex-Code/1.0.43"}
# audience = "https://api.openai.com/v1"
# user_agent = "Codex-Code/1.0.43"
# request_timeout = 30
# callback_timeout = 300
# callback_port = 1455
# use_pkce = true
```

## Related Components
- `provider.py`: OAuth implementation for Codex credentials
- `plugin.py`: runtime lifecycle for the auth plugin
- `config.py`: settings model for OAuth parameters
