"""Max tokens plugin for automatic token limit enforcement.

This plugin intercepts requests and automatically sets max_tokens based on
model limits from the pricing data when no max_tokens is provided.
"""

from .adapter import MaxTokensAdapter
from .config import MaxTokensConfig
from .plugin import factory


__all__ = ["MaxTokensAdapter", "MaxTokensConfig", "factory"]
