# Claude SDK Plugin

Runs Claude through the local Claude Code SDK and CLI with session management.

## Highlights
- Wraps the SDK adapter with reusable session and streaming support
- Validates CLI availability via the detection service and refresh task
- Shares the Claude accumulator to emit streaming metrics comparable to the API

## Configuration
- `ClaudeSDKSettings` covers CLI discovery, auth, and session pooling options
- Requires the Claude CLI to be installed and reachable on `PATH`
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin claude_sdk --config-class ClaudeSDKSettings`

```toml
[plugins.claude_sdk]
# enabled = true
# base_url = "claude-sdk://local"
# session_pool_enabled = false
# session_pool_size = 5
# include_system_messages_in_stream = true
# sdk_message_mode = "formatted"

[plugins.claude_sdk.sdk_session_pool]
# enabled = true
# session_ttl = 3600
# max_sessions = 1000
# cleanup_interval = 300
```

## Related Components
- `adapter.py`: bridge between CCProxy requests and the SDK client
- `tasks.py`: periodic detection refresh for CLI state
- `routes.py`: FastAPI router served under `/claude/sdk`
