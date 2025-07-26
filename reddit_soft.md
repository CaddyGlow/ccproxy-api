# Use your Claude Max subscription as an API - introducing ccproxy-api

**TL;DR:** ccproxy-api lets you use your Claude Max subscription ($20/month) instead of paying for API access. Run locally as a proxy server with both SDK mode (access to Claude tools) and API mode (direct proxy).

## What is ccproxy-api?

`ccproxy` is a local reverse proxy server that lets you use your existing Claude Max subscription to interact with the Anthropic API, bypassing the need for separate API key billing.

**GitHub:** [https://github.com/CaddyGlow/ccproxy-api/](https://github.com/CaddyGlow/ccproxy-api/)

## Key Features

- Use your $20/month Claude Max instead of paying per token
- Two modes: SDK mode (with Claude tools) or direct API mode
- Works with aider, aichat, and any Anthropic/OpenAI-compatible clients
- Simple installation via pipx or uvx

## How it works

The server provides two modes:

1. **SDK Mode (`/sdk`)** - Routes through Claude Code CLI, giving you access to tools like file editing, web search, etc.
2. **API Mode (`/api`)** - Direct reverse proxy with auth injection for full API access

## Quick Start

### Installation
```bash
# Run directly
pipx run ccproxy-api
# OR
uvx ccproxy-api

# Or install globally
pipx install ccproxy-api
```

### API Mode (No Claude Code needed)
```bash
# Login once
ccproxy-api auth login

# Start the proxy
uvx ccproxy-api

# Use with aider with OpenAI compatibility
OPENAI_API_KEY=dummy OPENAI_BASE_URL=http://127.0.0.1:8000/api/ \
  aider --model claude-sonnet-4-20250514
```

### SDK Mode (Requires Claude Code)
```bash
# Install Claude Code first
npm install -g @anthropic-ai/claude-code
claude /login

# Start with working directory and permissions
ccproxy serve --cwd /tmp/myproject --allowed-tools Read,Write --permission-mode acceptEdits

# Use with your favorite tool
aichat "Create a hello world in Python"
```

## Example Usage

Using it with aider in API mode:

```sh
$ ANTHROPIC_API_KEY=dummy ANTHROPIC_BASE_URL=http://127.0.0.1:8000/api aider --model claude-sonnet-4-20250514
───────────────────────────────────────────────────────────────────────────────
Aider v0.85.2
Main model: claude-sonnet-4-20250514 with diff edit format, infinite output
───────────────────────────────────────────────────────────────────────────────
> Hello Claude!

Hello! I'm Claude, ready to help you with your software development tasks.
I can help you create new files, modify existing code, or work on any
programming tasks you have in mind!

Tokens: 2.5k sent, 80 received. Cost: $0.0088 message, $0.0088 session.
```

## Current Limitations

- SDK mode output can have formatting issues with some clients
- Need to configure permissions for certain operations in SDK mode
- Claude Code must be installed separately for SDK features

## OpenAI Compatibility

The proxy translates OpenAI model names, so you can use `openai/o3` and it will map to Claude:

```bash
aider --model openai/o3  # Actually uses Claude
```

## More Information

Full documentation and examples available at [https://github.com/CaddyGlow/ccproxy-api/](https://github.com/CaddyGlow/ccproxy-api/)

Built this because I was paying for both Claude Max and API access. Figured others might find it useful.
