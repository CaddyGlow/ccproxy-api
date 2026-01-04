# Permissions Plugin

Provides interactive approval flows for tool calls and other privileged actions.

## Highlights
- Starts the permission service that tracks and resolves pending requests
- Exposes SSE and MCP routes for UI, terminal, or IDE integrations
- Supports configurable timeouts and optional terminal UI prompts

## Configuration
- `PermissionsConfig` toggles enablement, stream support, and timeouts
- Pending requests are handled only when the plugin is enabled
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin permissions --config-class PermissionsConfig`

```toml
[plugins.permissions]
# enabled = true
# timeout_seconds = 30
# enable_terminal_ui = true
# enable_sse_stream = true
# cleanup_after_minutes = 5
```

## Related Components
- `service.py`: permission service entrypoint
- `routes.py`: FastAPI router for SSE streaming
- `mcp/`: MCP server routes used by Claude Code
