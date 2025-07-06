# Configuration

Configure Claude Code Proxy API Server for your personal local setup and preferences.

## Configuration Methods

The server supports multiple configuration methods with the following priority order:

1. **Environment Variables** (highest priority)
2. **TOML Configuration Files** (`.ccproxy.toml`, `ccproxy.toml`, or `~/.config/ccproxy/config.toml`)
3. **JSON Configuration File** (`config.json`)
4. **Default Values** (lowest priority)

## Environment Variables

### Server Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PORT` | Server port | `8000` | `PORT=8080` |
| `HOST` | Server host | `0.0.0.0` | `HOST=127.0.0.1` |
| `LOG_LEVEL` | Logging level | `INFO` | `LOG_LEVEL=DEBUG` |

### Security Configuration

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `AUTH_TOKEN` | Authentication token for API access | None | `AUTH_TOKEN=abc123xyz789...` |

The proxy accepts authentication tokens in multiple header formats:
- **Anthropic Format**: `x-api-key: <token>` (takes precedence)
- **OpenAI/Bearer Format**: `Authorization: Bearer <token>`

All formats use the same configured `AUTH_TOKEN` value.

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
AUTH_TOKEN=abc123xyz789abcdef...  # Optional authentication
CLAUDE_CLI_PATH=/opt/claude/bin/claude
```

## TOML Configuration (Recommended)

TOML configuration files provide a more readable and structured format. Files are searched in this order:

1. `.ccproxy.toml` in the current directory
2. `ccproxy.toml` in the git repository root
3. `config.toml` in `~/.config/ccproxy/`

### Example TOML Configuration

```toml
# Server settings
host = "127.0.0.1"
port = 8080
log_level = "DEBUG"
workers = 2

# Security settings
cors_origins = ["https://example.com", "https://app.com"]
auth_token = "your-auth-token"

# Docker settings
[docker_settings]
docker_image = "custom-claude-image"
docker_volumes = ["/host/data:/container/data"]
docker_environment = {CLAUDE_ENV = "production"}

# Connection pool settings
[pool_settings]
enabled = true               # Enable/disable connection pooling
min_size = 2                # Minimum number of instances to maintain
max_size = 10               # Maximum number of instances allowed
idle_timeout = 300          # Seconds before idle connections are closed
warmup_on_startup = true    # Pre-create minimum instances on startup
health_check_interval = 60  # Seconds between connection health checks
acquire_timeout = 5.0       # Maximum seconds to wait for an available instance

# Claude Code options
[claude_code_options]
model = "claude-3-5-sonnet-20241022"
max_thinking_tokens = 30000
```

## JSON Configuration File

Create a `config.json` file in the project root for advanced configuration:

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "reload": false
  },
  "security": {
    "auth_token": "your-secure-token-here"
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

### Connection Pool Configuration

For improved performance, see the [Connection Pool Configuration Guide](/user-guide/pool-configuration/) for detailed settings.

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

### Security Configuration

Configure API authentication and security features:

```json
{
  "security": {
    "auth_token": "your-secure-token-here",    // Authentication token for API access
    "enabled": true                            // Enable/disable auth
  }
}
```

**Authentication Headers:** The proxy accepts tokens in multiple formats:
- **Anthropic Format**: `x-api-key: <token>` (takes precedence)
- **OpenAI/Bearer Format**: `Authorization: Bearer <token>`

All formats use the same configured `auth_token` value.

**Note:** When `auth_token` is not set or is null, authentication is disabled.

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

## Docker Configuration for Personal Use

### Environment Variables

```yaml
version: '3.8'
services:
  claude-code-proxy-api:
    image: claude-code-proxy
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
      - HOST=0.0.0.0
      - LOG_LEVEL=INFO
      - CLAUDE_CLI_PATH=/usr/local/bin/claude
    volumes:
      - ~/.config/claude:/root/.config/claude:ro
```

### Volume Mounting for Personal Setup

Mount your Claude configuration and local settings:

```yaml
version: '3.8'
services:
  claude-code-proxy-api:
    image: claude-code-proxy
    ports:
      - "8000:8000"
    volumes:
      - ./config.json:/app/config.json:ro
      - ./logs:/app/logs
      - ~/.config/claude:/root/.config/claude:ro
```

## Personal Use Configuration

### Recommended Settings for Local Development

```json
{
  "server": {
    "host": "127.0.0.1",
    "port": 8000,
    "workers": 2,
    "reload": true,
    "access_log": true,
    "proxy_headers": false
  },
  "logging": {
    "level": "INFO",
    "format": "text",
    "file": "./logs/app.log",
    "rotation": "1 day",
    "retention": "7 days"
  },

  "cors": {
    "enabled": true,
    "allow_origins": ["http://localhost:*", "http://127.0.0.1:*"],
    "allow_credentials": false
  },
  "health": {
    "check_claude_cli": true,
    "detailed_response": true
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

## Personal Use Scenarios

### Development & Testing

```json
{
  "server": {
    "host": "127.0.0.1",
    "reload": true,
    "workers": 1
  },
  "logging": {
    "level": "DEBUG",
    "format": "text"
  },

}
```

### Daily Personal Use

```json
{
  "server": {
    "host": "127.0.0.1",
    "reload": false,
    "workers": 2
  },
  "logging": {
    "level": "INFO",
    "format": "text"
  },

}
```

### Isolated Docker Setup

```json
{
  "server": {
    "host": "0.0.0.0",
    "reload": false,
    "workers": 2
  },
  "logging": {
    "level": "INFO",
    "format": "json",
    "file": "/app/logs/app.log"
  },

}
```

## Configuration Best Practices for Personal Use

1. **Use environment variables** for local customization and preferences
2. **Use configuration files** for structured settings you want to persist
3. **Validate configuration** before starting the server
4. **Keep backups** of your working configuration files
5. **Use different configurations** for development vs. daily use
6. **Start simple** - use defaults first, then customize as needed
7. **Secure your setup** - bind to localhost (127.0.0.1) for local-only access

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



## Advanced Configuration Reference

### Complete Configuration Options

#### .env File Reference

```bash
# Basic server configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
WORKERS=4
RELOAD=false

# Claude configuration
CLAUDE_CLI_PATH=/usr/local/bin/claude

# Security settings
AUTH_TOKEN=your-secure-token-here
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Tools handling
TOOLS_HANDLING=warning

# Security settings for subprocess execution
CLAUDE_USER=claude
CLAUDE_GROUP=claude
```

#### Complete JSON Configuration Schema

```json
{
  "host": "0.0.0.0",
  "port": 8000,
  "log_level": "INFO",
  "workers": 4,
  "reload": false,
  "cors_origins": ["https://yourdomain.com"],
  "claude_cli_path": "/usr/local/bin/claude",
  "tools_handling": "warning",
  "docker_settings": {
    "docker_image": "claude-code-proxy:latest",
    "docker_volumes": [
      "$HOME/.config/claude:/data/home",
      "$PWD:/data/workspace"
    ],
    "docker_environment": {
      "CLAUDE_HOME": "/data/home",
      "CLAUDE_WORKSPACE": "/data/workspace"
    },
    "docker_additional_args": ["--network=host"],
    "docker_home_directory": "/home/user/.config/claude",
    "docker_workspace_directory": "/home/user/projects"
  },
  "claude_code_options": {
    "cwd": "/path/to/working/directory",
    "model": "claude-3-5-sonnet-20241022",
    "max_thinking_tokens": 30000
  },
  "pool_settings": {
    "enabled": true,
    "min_size": 2,
    "max_size": 10,
    "idle_timeout": 300,
    "warmup_on_startup": true,
    "health_check_interval": 60,
    "acquire_timeout": 5.0
  }
}
```

### Configuration Validation

All configuration values are automatically validated:

- **Port**: Must be between 1-65535
- **Log Level**: Must be DEBUG, INFO, WARNING, ERROR, or CRITICAL
- **CORS Origins**: Must be valid URLs or "*"
- **Claude CLI Path**: Must exist and be executable
- **Tools Handling**: Must be "error", "warning", or "ignore"

### Environment-Specific Configuration Files

#### Development Environment (`.env.development`)
```bash
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=DEBUG
RELOAD=true
WORKERS=1
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

#### Production Environment (`.env.production`)
```bash
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
RELOAD=false
WORKERS=4
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com
CLAUDE_CLI_PATH=/usr/local/bin/claude
TOOLS_HANDLING=error
CLAUDE_USER=claude
CLAUDE_GROUP=claude
```

### Advanced Configuration Patterns

#### Configuration with Environment Variable Substitution

```json
{
  "host": "${HOST:-0.0.0.0}",
  "port": "${PORT:-8000}",
  "claude_cli_path": "${CLAUDE_CLI_PATH}",
  "cors_origins": ["${CORS_ORIGIN:-*}"]
}
```

#### Multi-Environment Loading Script

```bash
#!/bin/bash
# scripts/load-env.sh

ENV=${1:-development}

case $ENV in
  "development")
    export $(cat .env.development | xargs)
    ;;
  "production")
    export $(cat .env.production | xargs)
    ;;
  *)
    echo "Unknown environment: $ENV"
    exit 1
    ;;
esac

echo "Loaded configuration for: $ENV"
```

### CLI Configuration Commands

```bash
# Display current configuration
ccproxy config

# Test Claude CLI integration
ccproxy claude -- --version

# Test with Docker
ccproxy claude --docker -- --version

# Specify custom configuration file
CONFIG_FILE=/path/to/custom/config.json ccproxy run
```

### Advanced Troubleshooting

#### Configuration Debugging

```bash
# Enable debug logging to see configuration loading
LOG_LEVEL=DEBUG ccproxy run

# Validate configuration without starting server
python -c "from claude_code_proxy.config.settings import get_settings; print('Config valid')"

# Check Claude CLI path resolution
ccproxy claude -- --version
```

#### Common Advanced Issues

1. **Docker Volume Mount Issues**
   ```bash
   # Check volume permissions
   ls -la ~/.config/claude/
   
   # Fix permissions if needed
   chmod -R 755 ~/.config/claude/
   ```

2. **Environment Variable Substitution**
   ```bash
   # Test variable expansion
   echo "Host: ${HOST:-0.0.0.0}"
   echo "Port: ${PORT:-8000}"
   ```

3. **Complex CORS Configuration**
   ```bash
   # Multiple origins
   CORS_ORIGINS="https://app1.example.com,https://app2.example.com"
   
   # Development with multiple ports
   CORS_ORIGINS="http://localhost:3000,http://localhost:3001,http://127.0.0.1:3000"
   ```