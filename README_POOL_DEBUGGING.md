# Connection Pool Debug Logging

This document explains how to verify that the connection pool is working correctly through debug logs.

## Quick Start

To see pool activity, just watch the server logs while making requests:

```bash
# Terminal 1: Start the server
uv run python main.py

# Terminal 2: Make requests
python examples/test_pool_logging.py
# or
python examples/pool_performance_test.py
```

## Key Log Messages

### Startup
When the server starts with pooling enabled:
```
[STARTUP] Configuring connection pool manager...
[POOL_MANAGER] Claude instance pool configured (min: 2, max: 10)
[POOL] Initializing Claude instance pool (min=2, max=10)
[POOL] Created new pooled connection abc123
[POOL] Created new pooled connection def456
[POOL] Claude instance pool initialized successfully with 2 connections
[STARTUP] Claude connection pool initialized successfully
```

### Request Handling
When a request uses the pool:
```
[API] Acquiring Claude client from pool for message request
[POOL_MANAGER] Acquiring client from pool
[POOL] Reusing existing connection abc123 (use_count: 1, pool_size: 2)
[POOL_MANAGER] Acquired pooled connection abc123 (use_count: 1)
...
[API] Releasing Claude client connection abc123 back to pool
[POOL_MANAGER] Releasing connection abc123 back to pool
[POOL] Released connection abc123 back to pool (available: 2, in_use: 0)
```

The key indicator is **"Reusing existing connection"** - this means the pool is working!

### New Connection Creation
When all connections are busy:
```
[POOL] Created new connection ghi789 (total: 3/10)
```

### Pool Statistics
Check real-time stats:
```bash
curl http://localhost:8000/pool/stats | jq
```

Response:
```json
{
  "pool_enabled": true,
  "stats": {
    "connections_created": 5,
    "connections_destroyed": 2,
    "connections_reused": 150,  // High number = pool working well!
    "acquire_timeouts": 0,
    "health_check_failures": 0,
    "total_connections": 3,
    "available_connections": 2,
    "in_use_connections": 1
  }
}
```

## Performance Comparison

Run the performance test to see the speed improvement:

```bash
# With pooling (default)
python examples/pool_performance_test.py

# Without pooling
POOL_SETTINGS__ENABLED=false python examples/pool_performance_test.py
```

Typical results:
- First request: ~2-3 seconds (creates new connection)
- Subsequent requests: ~0.5-1 second (reuses connection)
- Speed improvement: 50-80% faster with pooling

## Configuration

Adjust pool settings in `.ccproxy.toml`:
```toml
[pool_settings]
enabled = true
min_size = 2      # Pre-created connections
max_size = 10     # Maximum connections
idle_timeout = 300  # Seconds before cleanup
```

Or via environment variables (note the double underscore):
```bash
export POOL_SETTINGS__ENABLED=true
export POOL_SETTINGS__MIN_SIZE=5
export POOL_SETTINGS__MAX_SIZE=20
export POOL_SETTINGS__IDLE_TIMEOUT=300

# Or disable pooling entirely:
export POOL_SETTINGS__ENABLED=false
```

## Troubleshooting

### No "Reusing" Messages
- Check if pooling is enabled: `curl http://localhost:8000/pool/stats`
- Verify configuration: pool_settings.enabled = true

### Many "Created new connection" Messages
- Pool might be too small for your load
- Increase max_size in configuration

### Slow First Request
- Normal behavior - first request creates connection
- Enable warmup_on_startup to pre-create connections

### Memory Usage Concerns
- Each connection uses ~50-100MB
- Reduce max_size or idle_timeout if needed