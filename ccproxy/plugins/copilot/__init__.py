"""GitHub Copilot provider plugin for CCProxy.

This plugin provides OAuth authentication with GitHub and API proxying
capabilities for GitHub Copilot services, following the established patterns
from existing OAuth Claude and Codex plugins.
"""

from .plugin import CopilotPluginFactory, CopilotPluginRuntime, factory


__all__ = ["CopilotPluginFactory", "CopilotPluginRuntime", "factory"]
