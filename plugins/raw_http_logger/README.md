# Raw HTTP Logger Plugin

A system plugin for CCProxy that provides raw HTTP logging capabilities for debugging purposes.

## Features

- Logs raw HTTP requests and responses at the transport level
- Configurable logging for client and provider traffic
- Security-aware header filtering
- Body size limits to prevent excessive disk usage
- Path exclusion for sensitive endpoints

## Configuration

The plugin can be configured through the CCProxy settings file or environment variables.

### Via Settings File

Add to your `settings.yaml` or `settings.json`:

```yaml
plugins:
  raw_http_logger:
    enabled: true
    log_dir: "/tmp/ccproxy/raw"
    log_client_request: true
    log_client_response: true
    log_provider_request: true
    log_provider_response: true
    max_body_size: 10485760  # 10MB
    include_paths: []  # If empty, all paths are included
    exclude_paths:     # Takes precedence over include_paths
      - "/health"
      - "/metrics"
    exclude_headers:
      - "authorization"
      - "cookie"
      - "x-api-key"
```

### Via Environment Variables (Backward Compatibility)

For backward compatibility, the plugin still supports the original environment variable:

```bash
export CCPROXY_LOG_RAW_HTTP=true
export CCPROXY_RAW_LOG_DIR=/tmp/ccproxy/raw
```

## Log Files

When enabled, the plugin creates log files in the configured directory with the following naming pattern:

- `{request_id}_client_request.http` - Raw HTTP request from client
- `{request_id}_client_response.http` - Raw HTTP response to client
- `{request_id}_provider_request.http` - Raw HTTP request to provider
- `{request_id}_provider_response.http` - Raw HTTP response from provider

## Security Considerations

The plugin automatically filters sensitive headers to prevent logging credentials:
- `authorization`
- `cookie`
- `x-api-key`

These headers will appear as `[REDACTED]` in the logs.

## Performance Impact

- Minimal memory overhead (no buffering, direct streaming to disk)
- Async I/O for file operations
- Configurable body size limits
- Path exclusion to skip logging for high-frequency endpoints

## Path Filtering

The plugin supports both inclusive and exclusive path filtering:

### Include Paths (Whitelist)
When `include_paths` is specified, **only** those paths will be logged:

```yaml
plugins:
  raw_http_logger:
    enabled: true
    include_paths:
      - "/api/v1/messages"
      - "/api/v1/chat"
      - "/claude"
```

### Exclude Paths (Blacklist)
`exclude_paths` takes precedence over `include_paths`:

```yaml
plugins:
  raw_http_logger:
    enabled: true
    include_paths:
      - "/api"  # Include all API paths
    exclude_paths:
      - "/api/health"  # But exclude health endpoint
      - "/api/metrics" # And metrics endpoint
```

### Combined Example
```yaml
plugins:
  raw_http_logger:
    enabled: true
    include_paths:
      - "/api/v1"      # Only log v1 API calls
      - "/claude"      # And Claude SDK calls
    exclude_paths:
      - "/api/v1/health"  # But never log health checks
```

## Debugging Tips

1. Enable only the specific traffic you need to debug:
   - Set `log_client_request: false` if you only need provider traffic
   - Set `log_provider_response: false` if you only need request debugging

2. Use path filtering for focused debugging:
   - Use `include_paths` to focus on specific endpoints
   - Use `exclude_paths` to filter out noisy endpoints

3. Set appropriate body size limits:
   - Reduce `max_body_size` for high-traffic scenarios
   - Increase for debugging large payloads

## Example Usage

1. Enable the plugin in your configuration
2. Make requests through CCProxy
3. Check the log directory for raw HTTP files
4. Use tools like `cat`, `less`, or `curl` to inspect the raw HTTP data

```bash
# View a request
cat /tmp/ccproxy/raw/abc123_client_request.http

# Follow provider responses in real-time
tail -f /tmp/ccproxy/raw/*_provider_response.http
```
