"""MiniMax provider plugin for CCProxy.

This plugin adds first-class support for the MiniMax LLM API, exposing
OpenAI- and Anthropic-compatible endpoints backed by MiniMax's global and
regional deployments.
"""

from .plugin import MiniMaxFactory, MiniMaxRuntime, factory


__all__ = ["MiniMaxFactory", "MiniMaxRuntime", "factory"]
