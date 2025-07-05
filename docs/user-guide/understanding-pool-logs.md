# Understanding Connection Pool Logs

This guide helps you understand the debug logs related to the connection pool feature, so you can verify it's working correctly.

## Log Prefixes

The pool system uses specific prefixes to help you identify different components:

- `[STARTUP]` - Application startup events
- `[SHUTDOWN]` - Application shutdown events
- `[POOL]` - Core pool operations
- `[POOL_MANAGER]` - Pool manager operations
- `[API]` - API endpoint pool interactions

## Key Log Messages

### Startup Logs

When the server starts, you'll see:

```
[STARTUP] Configuring connection pool manager...
[POOL_MANAGER] Claude instance pool configured (min: 2, max: 10)
[POOL] Initializing Claude instance pool (min=2, max=10)
[POOL] Created new pooled connection abc123
[POOL] Created new pooled connection def456
[POOL] Claude instance pool initialized successfully with 2 connections
[STARTUP] Claude connection pool initialized successfully
```

This shows:
- Pool is being configured with your settings
- Minimum connections are pre-created on startup
- Each connection gets a unique ID

### Request Handling Logs

When a request arrives, you'll see:

```
[API] Acquiring Claude client from pool for message request
[POOL_MANAGER] Acquiring client from pool
[POOL] Reusing existing connection abc123 (use_count: 1, pool_size: 2)
[POOL_MANAGER] Acquired pooled connection abc123 (use_count: 1)
```

This shows:
- API endpoint requests a client from the pool
- An existing connection is reused (not created new)
- Connection use count increases

After the request completes:

```
[API] Releasing Claude client connection abc123 back to pool
[POOL_MANAGER] Releasing connection abc123 back to pool
[POOL] Released connection abc123 back to pool (available: 2, in_use: 0)
```

### New Connection Creation

When all connections are busy:

```
[POOL] Created new connection ghi789 (total: 3/10)
[POOL_MANAGER] Acquired pooled connection ghi789 (use_count: 1)
```

This shows the pool expanding up to max_size when needed.

### Connection Cleanup

For idle connections:

```
[POOL] Cleaned up idle connection ghi789 (was idle for >=300 seconds)
[POOL] Destroyed connection ghi789 (total remaining: 2)
```

### Shutdown Logs

When the server shuts down:

```
[SHUTDOWN] Shutting down Claude connection pool...
[POOL] Shutting down Claude instance pool (destroying 2 connections)
[POOL] Destroyed connection abc123 (total remaining: 1)
[POOL] Destroyed connection def456 (total remaining: 0)
[POOL] Claude instance pool shutdown complete
[SHUTDOWN] Claude connection pool shut down successfully
```

## Verifying Pool Is Working

You know the pool is working correctly when you see:

1. **Connection Reuse**: Look for "Reusing existing connection" messages
2. **Use Count Increases**: The use_count value increases with each reuse
3. **No New Connections**: For sequential requests, no "Created new connection" messages
4. **Pool Statistics**: Check `/pool/stats` endpoint:

```bash
curl http://localhost:8000/pool/stats
```

Response shows pool activity:
```json
{
  "pool_enabled": true,
  "stats": {
    "connections_created": 5,
    "connections_destroyed": 2,
    "connections_reused": 150,  // High reuse count = pool working well
    "acquire_timeouts": 0,
    "health_check_failures": 0,
    "total_connections": 3,
    "available_connections": 2,
    "in_use_connections": 1
  }
}
```

## Common Scenarios

### Pool Disabled

If pooling is disabled:
```
[STARTUP] Claude connection pooling is disabled
[POOL_MANAGER] Creating new client (pooling disabled)
```

### Pool at Maximum

When pool reaches max_size and all connections are busy:
```
[POOL] Could not acquire Claude connection within 5.0s timeout
```

### Health Check Failures

If connections fail health checks:
```
[POOL] Health check failed for connection xyz789
[POOL] Destroyed unhealthy connection xyz789
```

## Troubleshooting with Logs

1. **No Reuse Messages**: Check if pool is enabled in configuration
2. **Many New Connections**: Pool might be too small for load
3. **Timeouts**: Increase max_size or acquire_timeout
4. **Frequent Destruction**: Check idle_timeout setting

## Example Test Script

Use the provided test script to see pool behavior:

```bash
python examples/test_pool_logging.py
```

Watch the server logs while the script runs to see:
- Connection reuse for sequential requests
- Pool expansion for concurrent requests
- Connection management throughout