# Configuration

Configure Claude Code Proxy API Server to meet your specific needs.

## Configuration Methods

The server supports multiple configuration methods with the following priority order:

1. **Environment Variables** (highest priority)
2. **Configuration File** (`config.json`)
3. **Default Values** (lowest priority)

## Environment Variables

### Server Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PORT` | Server port | `8000` | `PORT=8080` |
| `HOST` | Server host | `0.0.0.0` | `HOST=127.0.0.1` |
| `LOG_LEVEL` | Logging level | `INFO` | `LOG_LEVEL=DEBUG` |

### Claude CLI Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `CLAUDE_CLI_PATH` | Path to Claude CLI | Auto-detected | `CLAUDE_CLI_PATH=/usr/local/bin/claude` |

### Example Environment Setup

```bash
# .env file
PORT=8080
HOST=0.0.0.0
LOG_LEVEL=INFO
CLAUDE_CLI_PATH=/opt/claude/bin/claude
```

## Configuration File

Create a `config.json` file in the project root for advanced configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": false
  },
  "claude": {
    "cli_path": "/path/to/claude",
    "default_model": "claude-3-5-sonnet-20241022",
    "max_tokens": 4096,
    "timeout": 30
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "logs/app.log"
  },
  "rate_limiting": {
    "requests_per_minute": 60,
    "burst_size": 10,
    "enabled": true
  },
  "cors": {
    "enabled": true,
    "allow_origins": ["*"],
    "allow_methods": ["GET", "POST"],
    "allow_headers": ["*"]
  },
  "health": {
    "check_claude_cli": true,
    "detailed_response": false
  }
}
```

## Configuration Sections

### Server Configuration

Controls the FastAPI server behavior:

```json
{
  "server": {
    "host": "0.0.0.0",           // Bind address
    "port": 8000,                // Port number
    "workers": 4,                // Number of worker processes
    "reload": false,             // Auto-reload on file changes (dev only)
    "access_log": true,          // Enable access logging
    "proxy_headers": true        // Trust proxy headers
  }
}
```

### Claude Configuration

Controls Claude CLI integration:

```json
{
  "claude": {
    "cli_path": "/path/to/claude",              // Custom CLI path
    "default_model": "claude-3-5-sonnet-20241022", // Default model
    "max_tokens": 4096,                         // Default max tokens
    "timeout": 30,                              // Request timeout (seconds)
    "auto_detect_path": true,                   // Auto-detect CLI path
    "search_paths": [                           // Custom search paths
      "/usr/local/bin/claude",
      "/opt/claude/bin/claude"
    ]
  }
}
```

### Logging Configuration

Controls application logging:

```json
{
  "logging": {
    "level": "INFO",                    // Log level (DEBUG, INFO, WARNING, ERROR)
    "format": "json",                   // Log format (json, text)
    "file": "logs/app.log",            // Log file path (optional)
    "rotation": "1 day",               // Log rotation (optional)
    "retention": "30 days",            // Log retention (optional)
    "structured": true,                // Enable structured logging
    "include_request_id": true         // Include request IDs
  }
}
```

### Rate Limiting

Configure API rate limiting:

```json
{
  "rate_limiting": {
    "enabled": true,                   // Enable rate limiting
    "requests_per_minute": 60,         // Requests per minute per IP
    "burst_size": 10,                  // Burst allowance
    "storage": "memory",               // Storage backend (memory, redis)
    "redis_url": "redis://localhost:6379", // Redis URL (if using Redis)
    "exempt_ips": ["127.0.0.1"]       // IPs exempt from rate limiting
  }
}
```

### CORS Configuration

Configure Cross-Origin Resource Sharing:

```json
{
  "cors": {
    "enabled": true,                   // Enable CORS
    "allow_origins": ["*"],            // Allowed origins
    "allow_methods": ["GET", "POST"],  // Allowed methods
    "allow_headers": ["*"],            // Allowed headers
    "allow_credentials": false,        // Allow credentials
    "max_age": 86400                   // Preflight cache duration
  }
}
```

### Health Check Configuration

Configure health monitoring:

```json
{
  "health": {
    "check_claude_cli": true,          // Check Claude CLI availability
    "detailed_response": false,        // Include detailed health info
    "timeout": 5,                      // Health check timeout
    "include_version": true,           // Include version in response
    "include_metrics": false           // Include basic metrics
  }
}
```

## Claude CLI Auto-Detection

The server automatically searches for Claude CLI in these locations:

1. **Environment PATH**
2. **Common installation paths:**
   - `~/.claude/local/claude`
   - `~/node_modules/.bin/claude`
   - `./node_modules/.bin/claude`
   - `/usr/local/bin/claude`
   - `/opt/homebrew/bin/claude`
   - `/usr/bin/claude`

### Custom CLI Path

If Claude CLI is installed in a custom location:

```bash
# Environment variable
export CLAUDE_CLI_PATH=/custom/path/to/claude

# Configuration file
{
  "claude": {
    "cli_path": "/custom/path/to/claude"
  }
}
```

## Docker Configuration

### Environment Variables

```yaml
version: '3.8'
services:
  claude-proxy:
    image: claude-code-proxy
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - LOG_LEVEL=INFO
      - CLAUDE_CLI_PATH=/usr/local/bin/claude
```

### Volume Mounting

Mount configuration file:

```yaml
version: '3.8'
services:
  claude-proxy:
    image: claude-code-proxy
    ports:
      - "8000:8000"
    volumes:
      - ./config.json:/app/config.json:ro
      - ./logs:/app/logs
```

## Production Configuration

### Recommended Settings

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": false,
    "access_log": false,
    "proxy_headers": true
  },
  "logging": {
    "level": "WARNING",
    "format": "json",
    "file": "/var/log/claude-proxy/app.log",
    "rotation": "1 day",
    "retention": "30 days"
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 100,
    "burst_size": 20,
    "storage": "redis",
    "redis_url": "redis://redis:6379"
  },
  "cors": {
    "enabled": true,
    "allow_origins": ["https://yourdomain.com"],
    "allow_credentials": false
  },
  "health": {
    "check_claude_cli": true,
    "detailed_response": false
  }
}
```

## Configuration Validation

The server validates configuration on startup and will report errors for:

- Invalid port numbers
- Missing Claude CLI binary
- Invalid log levels
- Malformed JSON configuration
- Network binding issues

### Validation Example

```bash
# Check configuration without starting server
uv run python -m claude_code_proxy.config.validate config.json
```

## Environment-Specific Configurations

### Development

```json
{
  "server": {
    "reload": true,
    "workers": 1
  },
  "logging": {
    "level": "DEBUG",
    "format": "text"
  },
  "rate_limiting": {
    "enabled": false
  }
}
```

### Staging

```json
{
  "server": {
    "reload": false,
    "workers": 2
  },
  "logging": {
    "level": "INFO",
    "format": "json"
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 30
  }
}
```

### Production

```json
{
  "server": {
    "reload": false,
    "workers": 4
  },
  "logging": {
    "level": "WARNING",
    "format": "json",
    "file": "/var/log/claude-proxy/app.log"
  },
  "rate_limiting": {
    "enabled": true,
    "requests_per_minute": 100,
    "storage": "redis"
  }
}
```

## Configuration Best Practices

1. **Use environment variables** for secrets and deployment-specific settings
2. **Use configuration files** for complex, structured settings
3. **Validate configuration** before deployment
4. **Log configuration changes** for audit purposes
5. **Use different configurations** for different environments
6. **Monitor configuration** for runtime changes
7. **Backup configuration files** as part of your deployment process

## Troubleshooting Configuration

### Common Issues

1. **Server won't bind to port**
   - Check if port is already in use: `netstat -an | grep :8000`
   - Try a different port: `PORT=8001`
   - Check firewall settings

2. **Claude CLI not found**
   - Verify installation: `claude --version`
   - Check PATH: `echo $PATH`
   - Set explicit path: `CLAUDE_CLI_PATH=/path/to/claude`

3. **Configuration file not loaded**
   - Check file exists: `ls -la config.json`
   - Validate JSON syntax: `python -m json.tool config.json`
   - Check file permissions: `chmod 644 config.json`

4. **Rate limiting not working**
   - Check Redis connection (if using Redis storage)
   - Verify rate limiting is enabled in configuration
   - Check logs for rate limiting messages