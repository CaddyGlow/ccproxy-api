# Systemd Service Setup

This guide explains how to set up Claude Code Proxy API as a systemd user service that starts automatically on user login.

## Quick Setup

Run the interactive setup script:

```bash
./scripts/setup-systemd.sh
```

The script will:
- Detect your `uv` installation
- Create a systemd user service
- Configure environment variables
- Enable auto-start on login (optional)
- Start the service immediately (optional)

## Manual Setup

### 1. Create Service File

Copy the template and customize it:

```bash
cp systemd/ccproxy.service.template ~/.config/systemd/user/ccproxy.service
```

Edit `~/.config/systemd/user/ccproxy.service` and replace the placeholders:
- `{{WORKING_DIR}}` - Project directory path
- `{{UV_PATH}}` - Path to uv executable
- `{{MAIN_PY_PATH}}` - Full path to main.py
- `{{USER_PATH}}` - Your PATH environment variable
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

Example with custom configuration:

```ini
[Service]
Environment="PORT=8080"
Environment="LOG_LEVEL=DEBUG"
Environment="CCPROXY_CONFIG=/path/to/custom/config.toml"
```

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
   uv run python main.py
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
