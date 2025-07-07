"""Docker integration module for Claude Code Proxy.

This module provides a comprehensive Docker integration system with support for:
- Protocol-based adapter design for better testing and flexibility
- Enhanced error handling with contextual information
- Real-time output streaming with middleware support
- Comprehensive port publishing (including host interface binding)
- Unified path management using DockerPath
- User context management with proper UID/GID mapping
"""

from .adapter import DockerAdapter, create_docker_adapter
from .docker_path import DockerPath, DockerPathSet
from .middleware import (
    LoggerOutputMiddleware,
    create_chained_docker_middleware,
    create_logger_middleware,
)
from .models import DockerUserContext
from .protocol import (
    DockerAdapterProtocol,
    DockerEnv,
    DockerPortSpec,
    DockerResult,
    DockerVolume,
)
from .stream_process import (
    ChainedOutputMiddleware,
    DefaultOutputMiddleware,
    OutputMiddleware,
    ProcessResult,
    create_chained_middleware,
    run_command,
)
from .validators import create_docker_error, validate_port_spec


__all__ = [
    # Main adapter classes
    "DockerAdapter",
    "DockerAdapterProtocol",
    # Path management
    "DockerPath",
    "DockerPathSet",
    # User context
    "DockerUserContext",
    # Type aliases
    "DockerEnv",
    "DockerPortSpec",
    "DockerResult",
    "DockerVolume",
    # Streaming and middleware
    "OutputMiddleware",
    "DefaultOutputMiddleware",
    "ChainedOutputMiddleware",
    "LoggerOutputMiddleware",
    "ProcessResult",
    # Factory functions
    "create_docker_adapter",
    "create_docker_error",
    "create_logger_middleware",
    "create_chained_docker_middleware",
    "create_chained_middleware",
    # Utility functions
    "run_command",
    "validate_port_spec",
]
