"""Claude API provider plugin.

This plugin provides direct access to the Anthropic Claude API
with support for both native Anthropic format and OpenAI-compatible format.
"""

from .plugin import ClaudeAPIFactory, ClaudeAPIRuntime, factory


__all__ = ["ClaudeAPIFactory", "ClaudeAPIRuntime", "factory"]
