#!/usr/bin/env bash
# Build documentation with MkDocs

set -e

echo "Building Claude Code Proxy API documentation..."

# Check if docs dependencies are installed
if ! command -v mkdocs &>/dev/null; then
  echo "Installing documentation dependencies..."
  uv sync --group docs
fi

# Clean previous build
echo "Cleaning previous build..."
rm -rf site/

# Build documentation
echo "Building documentation..."
mkdocs build --clean --verbose

echo "Documentation built successfully!"
echo "Open 'site/index.html' in your browser to view the documentation."
