"""Command Replay Plugin - Generate curl and xh commands for provider requests."""

from .config import CommandReplayConfig
from .hook import CommandReplayHook
from .plugin import CommandReplayFactory, CommandReplayRuntime


# Export the factory for auto-discovery
factory = CommandReplayFactory()

__all__ = [
    "CommandReplayConfig",
    "CommandReplayHook",
    "CommandReplayRuntime",
    "CommandReplayFactory",
    "factory",
]
