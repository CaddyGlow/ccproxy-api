#!/bin/bash
# Setup script to install both Python and Node dependencies

set -e

echo "Installing Python dependencies..."
uv sync --all-extras --dev

# Check if claude command exists
if ! command -v claude &> /dev/null; then
    echo "Claude CLI not found. Attempting to install..."

    # Check for pnpm first, then npm
    if command -v pnpm &> /dev/null; then
        echo "Installing @anthropic-ai/claude-code with pnpm..."
        pnpm install -g @anthropic-ai/claude-code
    elif command -v npm &> /dev/null; then
        echo "Installing @anthropic-ai/claude-code with npm..."
        npm install -g @anthropic-ai/claude-code
    else
        echo "Error: Neither pnpm nor npm found. Please install Node.js and a package manager."
        exit 1
    fi
else
    echo "Claude CLI already installed at: $(which claude)"
fi

echo "Installing Node dependencies..."
if command -v pnpm &> /dev/null; then
    pnpm install
elif command -v npm &> /dev/null; then
    npm install
fi

echo "Installing pre-commit hooks..."
uv run pre-commit install

echo "Setup complete!"

# Verify claude installation
if command -v claude &> /dev/null; then
    echo "Claude CLI version: $(claude --version || echo 'version command not available')"
else
    echo "Warning: Claude CLI still not found. You may need to add it to your PATH."
fi
