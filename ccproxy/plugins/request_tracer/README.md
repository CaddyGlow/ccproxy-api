# Request Tracer Plugin

Captures detailed request and response traces for troubleshooting.

## Highlights
- Registers both the core HTTP tracer and contextual request hook handlers
- Writes JSON or raw HTTP logs with size and path filtering controls
- Validates configuration and reports issues through structured logs

## Configuration
- `RequestTracerConfig` toggles enablement, formats, filters, and log targets
- Hook registration occurs only when a hook registry is available
- Generate defaults with `python3 scripts/generate_config_from_model.py \
  --format toml --plugin request_tracer --config-class RequestTracerConfig`

```toml
[plugins.request_tracer]
# enabled = true
# verbose_api = true
# json_logs_enabled = true
# raw_http_enabled = true
# trace_oauth = true
# log_dir = "/tmp/ccproxy/traces"
# exclude_paths = ["/health", "/metrics", "/readyz", "/livez"]
# include_paths = []
# exclude_headers = ["authorization", "x-api-key", "cookie", "x-auth-token"]
# redact_sensitive = true
# max_body_size = 10485760
# truncate_body_preview = 1024
# log_client_request = true
# log_client_response = true
# log_provider_request = true
# log_provider_response = true
# log_streaming_chunks = false
```

## Related Components
- `hook.py`: handles request lifecycle events and formatting
- `plugin.py`: runtime validation and hook wiring
- `config.py`: settings model for log output options
