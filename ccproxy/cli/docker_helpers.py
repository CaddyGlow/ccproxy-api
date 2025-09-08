"""Docker helpers for CLI commands using plugin system."""

from typing import TYPE_CHECKING, Any

from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    # from plugins.docker.config import DockerConfig
    # from plugins.docker.service import DockerService
    # Docker plugin not available - using Any for type checking
    from typing import Any as DockerConfig
    from typing import Any as DockerService


def get_docker_service() -> "DockerService | None":
    """Get Docker service from plugin registry if available.

    Returns:
        Docker service instance or None if Docker plugin not available
    """
    try:
        # Try to get service container and plugin registry
        # This is a simplified approach since the plugin registry
        # integration is complex and may not be fully initialized during CLI usage
        return None
    except (ImportError, AttributeError):
        return None


def create_docker_adapter() -> Any:
    """Create Docker adapter, either from plugin or fallback to direct import.

    Returns:
        Docker adapter instance

    Raises:
        RuntimeError: If Docker is not available
    """
    # Try to get Docker service from plugin first
    docker_service = get_docker_service()
    if docker_service and docker_service.is_enabled():
        return docker_service.adapter

    # Fallback to importing from plugin directly
    # try:
    #     from plugins.docker.service import create_docker_adapter as plugin_create
    #
    #     return plugin_create()
    # except ImportError as e:
    raise RuntimeError(
        "Docker functionality not available. Docker plugin is not installed."
    )


def get_docker_config() -> "DockerConfig | None":
    """Get Docker configuration from plugin registry if available.

    Returns:
        Docker config instance or None if Docker plugin not available
    """
    docker_service = get_docker_service()
    if docker_service:
        return docker_service.config
    return None


def get_docker_config_with_fallback(settings: Settings) -> "DockerConfig":
    """Get Docker configuration with fallback to settings.docker.

    Args:
        settings: Main application settings

    Returns:
        Docker configuration
    """
    # Try to get config from plugin first
    docker_config = get_docker_config()
    if docker_config:
        return docker_config

    # Fallback to creating config from legacy settings (if they exist)
    # Docker plugin not available, return None
    # try:
    #     from plugins.docker.config import DockerConfig
    #     ...
    # except ImportError:
    #     ...
    raise RuntimeError(
        "Docker functionality not available. Docker plugin is not installed."
    )


def is_docker_available() -> bool:
    """Check if Docker functionality is available.

    Returns:
        True if Docker is available, False otherwise
    """
    try:
        create_docker_adapter()
        return True
    except RuntimeError:
        return False
