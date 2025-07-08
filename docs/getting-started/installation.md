# Installation

## Prerequisites

- Python 3.11 or higher
- Claude account with active subscription (Pro, Team, or Enterprise)

## Installation Methods

### Using pipx (Recommended)

The easiest way to install ccproxy is using pipx, which installs it in an isolated environment:

```bash
# Install pipx if you haven't already
python -m pip install --user pipx
python -m pipx ensurepath

# Install ccproxy
pipx install git+https://github.com/CaddyGlow/claude-code-proxy-api.git
```

### From Source (Development)

```bash
git clone https://github.com/CaddyGlow/claude-code-proxy-api.git
cd claude-code-proxy-api

# Using uv (recommended for development)
uv sync

# Or using pip
pip install -e .
```

## Authentication Setup

After installation, you need to authenticate with Claude:

### Initial Login

```bash
# Login to Claude (opens browser)
ccproxy auth login
```

This will open a browser window for Claude authentication. Your credentials are securely stored locally.

### Verify Authentication

```bash
# Check credential status
ccproxy auth validate
```

Example output:
```
             Claude Credentials Validation

                     Credential Status
╭──────────────┬───────────────────────────────────────────╮
│ Property     │ Value                                     │
├──────────────┼───────────────────────────────────────────┤
│ Status       │ Valid                                     │
│ Subscription │ max                                       │
│ Expires      │ 2025-07-08 17:13:05 UTC (0d 5h remaining) │
│ Scopes       │ user:inference, user:profile              │
╰──────────────┴───────────────────────────────────────────╯
   success   ✓ Valid Claude credentials found
```

### View Credential Details

```bash
# Get detailed information (auto-renews if expired)
ccproxy auth info
```

This displays full credential details and automatically renews the token if expired.

## Configuration

### Option 1: Local Installation

After authentication, test the installation:
ccproxy claude -- /status
```

### Option 2: Docker (Recommended)

Use Docker with included Claude CLI (no local installation needed):

**Volume Configuration:**
- **Claude Home**: `~/.config/cc-proxy/home` (isolated from local Claude config)
- **Working Directory**: Current user path (same as local execution)

**Authentication:**
```bash
# Authenticate Claude in Docker (first time setup)
ccproxy claude --docker -- auth login
```

**Verification:**
```bash
# Test Docker Claude CLI
ccproxy claude --docker -- /status
```

### Expected Output

For both options, `ccproxy claude -- /status` or `ccproxy claude --docker -- /status` should show:

```
Executing: /path/to/claude /status

╭─────────────────────────────────────────────────────────╮
│ ✻ Welcome to Claude Code!                               │
│                                                         │
│   /help for help, /status for your current setup        │
╰─────────────────────────────────────────────────────────╯

 Claude Code Status v1.0.43

 Account • /login
  L Login Method: Claude Max Account  
  L Organization: your-email@example.com's Organization
  L Email: your-email@example.com

 Model • /model
  L sonnet (claude-sonnet-4-20250514)
```

**This confirms:**
- Claude CLI is properly installed and accessible (local or Docker)
- Your authentication is working
- The proxy can detect and use Claude CLI
- **Docker**: Uses isolated config at `~/.config/cc-proxy/home`

## Quick Start

Run the server:

```bash
claude-code-proxy-api
```

The server will start on `http://localhost:8000` by default.
