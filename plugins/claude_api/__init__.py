"""Claude API provider plugin.

This plugin provides direct access to the Anthropic Claude API
with support for both native Anthropic format and OpenAI-compatible format.
"""

from .plugin import factory


__all__ = ["factory"]
