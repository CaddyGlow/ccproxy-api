# Access Log Plugin

Captures client and provider traffic and writes structured access logs.

## Highlights
- Hooks into request lifecycle events to log Common, Combined, or JSON formats
- Supports per-channel enablement and configurable log destinations
- Optionally forwards records to the analytics ingest service for persistence

## Configuration
- `AccessLogConfig` controls enable flags, formats, and file paths
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin access_log`

```toml
[plugins.access_log]
# enabled = true
# client_enabled = true
# client_format = "structured"
# client_log_file = "/tmp/ccproxy/access.log"
# provider_enabled = false
# provider_format = "structured"
# provider_log_file = "/tmp/ccproxy/provider_access.log"
# exclude_paths = ["/health", "/metrics", "/readyz", "/livez"]
# buffer_size = 100
# flush_interval = 1.0
```

## Related Components
- `plugin.py`: runtime lifecycle and hook registration
- `hook.py`: event handler that formats and writes log entries
- `formatter.py` / `writer.py`: formatting helpers and async file writer
