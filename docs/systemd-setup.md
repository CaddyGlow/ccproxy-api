# Systemd Service Setup

This guide explains how to set up Claude Code Proxy API as a systemd user service that starts automatically on user login.

## Quick Setup

Run the interactive setup script:

```bash
./scripts/setup-systemd.sh
```

The script will:
- Detect your `uv` installation
- Ask for your working directory (defaults to a temp folder)
- Automatically configure the service with UV_PROJECT pointing to the code
- Include ~/.local/bin in PATH if it exists
- Enable auto-start on login (optional)
- Start the service immediately (optional)

**Note:** The setup is now simplified - you only need to specify the working directory where the service will run. Everything else is configured automatically.

## Manual Setup

### 1. Create Service File

Copy the template and customize it:

```bash
cp systemd/ccproxy.service.template ~/.config/systemd/user/ccproxy.service
```

Edit `~/.config/systemd/user/ccproxy.service` and replace the placeholders:
- `{{WORKING_DIR}}` - Working directory where the service runs (can be any directory)
- `{{UV_PATH}}` - Path to uv executable
- `{{UV_PROJECT}}` - Path to the Claude Code Proxy API project directory
- `{{USER_PATH}}` - Your PATH environment variable (should include ~/.local/bin if it exists)
- `{{USER_HOME}}` - Your home directory
- `{{EXTRA_ENV}}` - Additional environment variables (optional)

### 2. Enable and Start Service

```bash
# Reload systemd daemon
systemctl --user daemon-reload

# Enable service to start on login
systemctl --user enable ccproxy.service

# Start service now
systemctl --user start ccproxy.service

# Check status
systemctl --user status ccproxy.service
```

## Service Management

### Common Commands

```bash
# Start service
systemctl --user start ccproxy

# Stop service
systemctl --user stop ccproxy

# Restart service
systemctl --user restart ccproxy

# Check status
systemctl --user status ccproxy

# View logs
journalctl --user -u ccproxy -f

# Enable auto-start
systemctl --user enable ccproxy

# Disable auto-start
systemctl --user disable ccproxy
```

### Configuration

The service can be configured using:
1. Environment variables in the service file
2. `.ccproxy.toml` configuration file
3. Command-line arguments in `ExecStart`

#### Environment Variable Configuration

You can set environment variables directly in the service file using the `Environment=` directive:

```ini
[Service]
# Basic configuration
Environment="PORT=8080"
Environment="HOST=0.0.0.0"
Environment="LOG_LEVEL=DEBUG"
Environment="AUTH_TOKEN=your-secure-token"

# Using nested syntax
Environment="SERVER__PORT=8080"
Environment="SERVER__LOG_LEVEL=INFO"
Environment="SERVER__LOG_FORMAT=json"
Environment="SECURITY__AUTH_TOKEN=your-secure-token"

# Special environment variables
Environment="CONFIG_FILE=/etc/ccproxy/config.toml"
Environment="CCPROXY_VERBOSE_API=true"
Environment="CCPROXY_JSON_LOGS=true"

# Scheduler and pricing
Environment="SCHEDULER__ENABLED=true"
Environment="SCHEDULER__PRICING_UPDATE_INTERVAL_HOURS=24"
Environment="PRICING__UPDATE_ON_STARTUP=true"
```

#### Using an Environment File

For many environment variables, use an `EnvironmentFile`:

```ini
[Service]
EnvironmentFile=/etc/ccproxy/environment
```

Create `/etc/ccproxy/environment`:
```bash
# Server configuration
SERVER__HOST=0.0.0.0
SERVER__PORT=8000
SERVER__LOG_LEVEL=INFO
SERVER__LOG_FORMAT=json
SERVER__LOG_FILE=/var/log/ccproxy/app.log

# Security
SECURITY__AUTH_TOKEN=your-secure-token

# Claude configuration
CLAUDE__CLI_PATH=/usr/local/bin/claude

# Scheduler settings
SCHEDULER__ENABLED=true
SCHEDULER__MAX_CONCURRENT_TASKS=10
SCHEDULER__PRICING_UPDATE_INTERVAL_HOURS=24

# Pricing settings
PRICING__UPDATE_ON_STARTUP=true
PRICING__ENABLE_COST_TRACKING=true

# CORS settings
CORS__ALLOW_ORIGINS=["https://yourdomain.com"]
```

#### Complete Service File Example

```ini
[Unit]
Description=Claude Code Proxy API Server
After=network.target

[Service]
Type=exec
WorkingDirectory=/tmp/ccproxy-workdir
ExecStart=/home/user/.local/bin/uv run ccproxy serve

# Environment configuration
Environment="PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin"
Environment="HOME=/home/user"
Environment="UV_PROJECT=/home/user/claude-code-proxy-api"

# Load additional environment from file
EnvironmentFile=-/etc/ccproxy/environment

# Optional: Override specific settings
Environment="SERVER__PORT=8080"
Environment="SERVER__LOG_LEVEL=INFO"

# Restart configuration
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

For a complete list of environment variables, see [CLAUDE.md](/CLAUDE.md).

## Troubleshooting

### Service Won't Start

1. Check logs for errors:
   ```bash
   journalctl --user -u ccproxy -e
   ```

2. Verify uv is accessible:
   ```bash
   which uv
   ```

3. Test manual startup:
   ```bash
   cd /path/to/ccproxy
   uv run ccproxy serve
   ```

### Permission Issues

Ensure the working directory and files are readable by your user:
```bash
ls -la /path/to/ccproxy
```

### Service Not Starting on Login

1. Check if user lingering is enabled:
   ```bash
   loginctl show-user $USER | grep Linger
   ```

2. Enable lingering if needed:
   ```bash
   sudo loginctl enable-linger $USER
   ```

## Security Considerations

- The service runs with your user privileges
- Store sensitive configuration in secure files with restricted permissions
- Consider using systemd's credential storage for API keys
- Review logs regularly for unauthorized access attempts

## Multiple Instances

To run multiple instances with different configurations:

1. Create separate service files with unique names
2. Use different ports for each instance
3. Configure each with its own config file

Example:
```bash
# Development instance
systemctl --user start ccproxy-dev

# Production instance
systemctl --user start ccproxy-prod
```
