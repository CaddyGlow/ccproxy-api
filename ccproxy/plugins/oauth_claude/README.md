# OAuth Claude Plugin

Standalone OAuth provider for managing Claude API access tokens.

## Highlights
- Implements `ClaudeOAuthProvider` for authorization-code and refresh flows
- Integrates with the auth runtime to expose a reusable auth manager
- Shares detection utilities to diagnose CLI and token availability

## Configuration
- `ClaudeOAuthConfig` sets client credentials, scopes, and storage paths
- Works alongside `claude_api` or credential balancer powered providers
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin oauth_claude --config-class ClaudeOAuthConfig`

```toml
[plugins.oauth_claude]
# enabled = true
# base_url = "https://console.anthropic.com"
# token_url = "https://console.anthropic.com/v1/oauth/token"
# authorize_url = "https://claude.ai/oauth/authorize"
# profile_url = "https://api.anthropic.com/api/oauth/profile"
# client_id = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
# redirect_uri = "http://localhost:35593/callback"
# scopes = ["org:create_api_key", "user:profile", "user:inference"]
# request_timeout = 30
# callback_timeout = 300
# callback_port = 35593
# use_pkce = true
```

## Related Components
- `provider.py`: OAuth implementation and token storage helpers
- `plugin.py`: runtime wiring for auth-only plugin
- `config.py`: settings model for Claude OAuth parameters
