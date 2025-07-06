# Health & Monitoring

Health check and monitoring endpoints for your Claude Code Proxy.

## Health Check

### GET /health

Basic health check endpoint to verify the proxy is running.

#### Request

```http
GET /health
```

#### Response (Healthy)

```json
{
  "status": "healthy",
  "service": "claude-proxy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0"
}
```

#### Response (Unhealthy)

```json
{
  "status": "unhealthy",
  "service": "claude-proxy",
  "timestamp": "2024-01-15T10:30:00Z",
  "error": "Claude CLI not available"
}
```

## Detailed Health Check

### GET /health/detailed

Comprehensive health check with component status.

#### Request

```http
GET /health/detailed
```

#### Response

```json
{
  "status": "healthy",
  "service": "claude-proxy",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "claude_cli": {
      "status": "healthy",
      "path": "/usr/local/bin/claude",
      "version": "2.0.0"
    },
    "api_server": {
      "status": "healthy",
      "uptime": "2h 15m 30s",
      "port": 8000
    },
    "authentication": {
      "status": "healthy",
      "logged_in": true
    }
  },
  "configuration": {
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": "INFO",
    "cors_enabled": true
  }
}
```

## Status Codes

| HTTP Status | Service Status | Description |
|-------------|----------------|-------------|
| 200 | healthy | All systems operational |
| 503 | unhealthy | Service unavailable |
| 500 | error | Internal error occurred |

## Component Status

### Claude CLI Status

Indicates if the Claude CLI is available and authenticated:

```json
{
  "claude_cli": {
    "status": "healthy|unhealthy|error",
    "path": "/path/to/claude",
    "version": "2.0.0",
    "authenticated": true,
    "last_check": "2024-01-15T10:30:00Z"
  }
}
```

### API Server Status

Shows server runtime information:

```json
{
  "api_server": {
    "status": "healthy",
    "uptime": "2h 15m 30s",
    "port": 8000,
    "host": "0.0.0.0",
    "requests_processed": 1250,
    "errors": 3
  }
}
```

### Authentication Status

Displays Claude CLI authentication state:

```json
{
  "authentication": {
    "status": "healthy|unauthenticated|error",
    "logged_in": true,
    "user": "user@example.com",
    "expires": "2024-02-15T10:30:00Z"
  }
}
```

## Monitoring Integration

### Prometheus Metrics

If monitoring is enabled, metrics are available at:

```http
GET /metrics
```

Common metrics include:
- `claude_proxy_requests_total` - Total requests processed
- `claude_proxy_request_duration_seconds` - Request duration histogram
- `claude_proxy_errors_total` - Total errors by type
- `claude_proxy_claude_cli_status` - Claude CLI availability (0/1)

### Health Check for Load Balancers

For simple load balancer health checks:

```bash
# Returns 200 if healthy, 503 if unhealthy
curl -f http://localhost:8000/health
```

### Kubernetes Readiness/Liveness

Example Kubernetes probe configuration:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health/detailed
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
```

## Troubleshooting

### Common Health Issues

#### Claude CLI Not Found

```json
{
  "status": "unhealthy",
  "error": "Claude CLI not found in PATH",
  "details": {
    "searched_paths": [
      "/usr/local/bin/claude",
      "/usr/bin/claude",
      "~/.local/bin/claude"
    ]
  }
}
```

**Solution**: Install Claude CLI or set `CLAUDE_CLI_PATH`

#### Authentication Failed

```json
{
  "status": "unhealthy",
  "error": "Claude CLI authentication failed",
  "details": {
    "claude_cli_status": "not_authenticated"
  }
}
```

**Solution**: Run `claude auth login`

#### Port Already in Use

```json
{
  "status": "unhealthy",
  "error": "Port 8000 already in use",
  "details": {
    "port": 8000,
    "host": "0.0.0.0"
  }
}
```

**Solution**: Change port with `--port` or `PORT` environment variable

## Usage Examples

### Basic Health Check

```bash
curl http://localhost:8000/health
```

### Health Check with Details

```bash
curl http://localhost:8000/health/detailed | jq
```

### Python Health Check

```python
import requests

def check_proxy_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("status") == "healthy"
        return False
    except requests.RequestException:
        return False

if check_proxy_health():
    print("Proxy is healthy")
else:
    print("Proxy is unhealthy")
```

### Monitoring Script

```bash
#!/bin/bash
# Simple monitoring script

PROXY_URL="http://localhost:8000"

while true; do
    if curl -f "$PROXY_URL/health" > /dev/null 2>&1; then
        echo "$(date): Proxy is healthy"
    else
        echo "$(date): Proxy is unhealthy - checking details..."
        curl "$PROXY_URL/health/detailed" 2>/dev/null | jq '.error // "Unknown error"'
    fi
    sleep 30
done
```
