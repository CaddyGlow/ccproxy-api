"""Docker helpers for CLI commands using plugin system."""

from typing import TYPE_CHECKING, Any

from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    from plugins.docker.config import DockerConfig
    from plugins.docker.service import DockerService


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
    try:
        from plugins.docker.service import create_docker_adapter as plugin_create

        return plugin_create()
    except ImportError as e:
        raise RuntimeError(
            "Docker functionality not available. "
            "Make sure Docker plugin is installed and enabled."
        ) from e


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
    try:
        from plugins.docker.config import DockerConfig

        # Try to access legacy settings.docker (may not exist)
        if hasattr(settings, "docker"):
            return DockerConfig(
                enabled=True,
                docker_image=settings.docker.docker_image,
                docker_volumes=settings.docker.docker_volumes,
                docker_environment=settings.docker.docker_environment,
                docker_additional_args=settings.docker.docker_additional_args,
                docker_home_directory=settings.docker.docker_home_directory,
                docker_workspace_directory=settings.docker.docker_workspace_directory,
                user_mapping_enabled=settings.docker.user_mapping_enabled,
                user_uid=settings.docker.user_uid,
                user_gid=settings.docker.user_gid,
            )
        else:
            # No legacy settings, create a basic config
            return DockerConfig()
    except ImportError:
        # If we can't import the plugin config, create a basic config
        from plugins.docker.config import DockerConfig

        return DockerConfig()


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
