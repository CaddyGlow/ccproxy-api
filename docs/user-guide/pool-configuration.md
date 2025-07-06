# Connection Pool Configuration

The Claude Code Proxy API Server includes a high-performance connection pool system to minimize latency and improve throughput when handling multiple requests.

## Overview

The connection pool maintains pre-initialized Claude client instances that can be reused across requests, eliminating the overhead of creating new subprocess connections for each API call. This can reduce response latency by 200-500ms per request.

## Configuration

The connection pool can be configured through TOML configuration files, environment variables, or JSON configuration.

### TOML Configuration

Add pool settings to your `.ccproxy.toml` or `ccproxy.toml` file:

```toml
[pool_settings]
enabled = true               # Enable/disable connection pooling
min_size = 2                # Minimum number of instances to maintain
max_size = 10               # Maximum number of instances allowed
idle_timeout = 300          # Seconds before idle connections are closed
warmup_on_startup = true    # Pre-create minimum instances on startup
health_check_interval = 60  # Seconds between connection health checks
acquire_timeout = 5.0       # Maximum seconds to wait for an available instance
```

### Environment Variables

You can also configure the pool using environment variables:

```bash
export POOL_SETTINGS__ENABLED=true
export POOL_SETTINGS__MIN_SIZE=2
export POOL_SETTINGS__MAX_SIZE=10
export POOL_SETTINGS__IDLE_TIMEOUT=300
export POOL_SETTINGS__WARMUP_ON_STARTUP=true
export POOL_SETTINGS__HEALTH_CHECK_INTERVAL=60
export POOL_SETTINGS__ACQUIRE_TIMEOUT=5.0
```

### Default Values

- `enabled`: `true` - Pool is enabled by default
- `min_size`: `2` - Maintains at least 2 connections
- `max_size`: `10` - Allows up to 10 concurrent connections
- `idle_timeout`: `300` - Connections idle for 5 minutes are closed
- `warmup_on_startup`: `true` - Pre-creates minimum connections on startup
- `health_check_interval`: `60` - Checks connection health every minute
- `acquire_timeout`: `5.0` - Waits up to 5 seconds for an available connection

## Pool Behavior

### Startup

When the server starts with pooling enabled:

1. The pool manager is configured with your settings
2. If `warmup_on_startup` is true, `min_size` connections are pre-created
3. Background tasks start for health checking and idle cleanup

### Request Handling

When an API request arrives:

1. The endpoint requests a client from the pool
2. If available, a healthy connection is returned immediately
3. If none available but under `max_size`, a new connection is created
4. If at `max_size` and none available, waits up to `acquire_timeout` seconds
5. After use, the connection is released back to the pool

### Health Management

The pool automatically:

- Validates connections before reuse
- Removes unhealthy connections
- Maintains at least `min_size` healthy connections
- Cleans up connections idle longer than `idle_timeout`

## Performance Tuning

### For High Traffic

Increase pool size for better concurrency:

```toml
[pool_settings]
min_size = 5
max_size = 20
idle_timeout = 600  # Keep connections longer
```

### For Low Traffic

Reduce resource usage:

```toml
[pool_settings]
min_size = 1
max_size = 5
idle_timeout = 120  # Close idle connections sooner
warmup_on_startup = false  # Don't pre-create connections
```

### Disable Pooling

If you experience issues or prefer fresh connections:

```toml
[pool_settings]
enabled = false
```

## Monitoring

Check pool statistics via the `/pool/stats` endpoint:

```bash
curl http://localhost:8000/pool/stats
```

Response:

```json
{
  "pool_enabled": true,
  "stats": {
    "connections_created": 10,
    "connections_destroyed": 3,
    "connections_reused": 150,
    "acquire_timeouts": 0,
    "health_check_failures": 1,
    "total_connections": 7,
    "available_connections": 5,
    "in_use_connections": 2
  }
}
```

## Troubleshooting

### High Acquire Timeouts

If you see many `acquire_timeouts`, consider:
- Increasing `max_size` to handle more concurrent requests
- Increasing `acquire_timeout` to wait longer
- Checking if requests are taking too long to complete

### Memory Usage

Each pooled connection maintains a Claude client instance. If memory is a concern:
- Reduce `max_size`
- Decrease `idle_timeout` to clean up unused connections faster
- Disable pooling entirely with `enabled = false`

### Connection Errors

If connections frequently fail health checks:
- Check Claude CLI installation and configuration
- Review server logs for specific error messages
- Consider increasing `health_check_interval` to reduce check frequency
