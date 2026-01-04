# Command Replay Plugin

Generates reproducible `curl` and `xh` commands for captured provider requests.

## Highlights
- Subscribes to provider hook events to snapshot raw HTTP payloads
- Emits commands to stdout or disk with configurable file layout
- Supports URL include/exclude filters and provider-only logging

## Configuration
- `CommandReplayConfig` toggles command types, log directory, and filters
- Enable via `plugins.command_replay` settings or matching environment vars
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin command_replay --config-class CommandReplayConfig`

```toml
[plugins.command_replay]
# enabled = true
# generate_curl = true
# generate_xh = true
# log_dir = "/tmp/ccproxy/command_replay"
# write_to_files = true
# separate_files_per_command = true
# include_url_patterns = ["api.anthropic.com", "api.openai.com", "claude.ai", "chatgpt.com"]
# exclude_url_patterns = []
# log_to_console = false
# log_level = "TRACE"
# only_provider_requests = false
```

## Related Components
- `hook.py`: assembles commands from hook payloads
- `formatter.py`: file naming and formatting helpers
- `plugin.py`: runtime wiring and hook registration
