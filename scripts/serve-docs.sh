#!/usr/bin/env bash
# Serve documentation locally with live reload

set -e

echo "Starting MkDocs development server..."

# Check if docs dependencies are installed
if ! command -v mkdocs &>/dev/null; then
  echo "Installing documentation dependencies..."
  uv sync --group docs
fi

# Start development server
echo "Serving documentation at http://127.0.0.1:8080"
echo "Press Ctrl+C to stop the server"

mkdocs serve --dev-addr 127.0.0.1:8080
