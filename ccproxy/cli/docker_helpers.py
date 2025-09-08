"""Docker helpers for CLI commands using plugin system."""

from typing import TYPE_CHECKING

from ccproxy.config.settings import Settings


if TYPE_CHECKING:
    from ccproxy.plugins.docker.adapter import DockerAdapter
    from ccproxy.plugins.docker.config import DockerConfig


def create_docker_adapter() -> "DockerAdapter":
    """Create Docker adapter from the Docker plugin.

    Returns:
        Docker adapter instance

    Raises:
        RuntimeError: If Docker is not available
    """
    try:
        from ccproxy.plugins.docker.adapter import DockerAdapter
        from ccproxy.plugins.docker.config import DockerConfig

        # Create adapter with default config
        config = DockerConfig()
        return DockerAdapter(config=config)

    except ImportError as e:
        raise RuntimeError(
            f"Docker plugin not available: {e}. Docker functionality is not installed."
        )


def get_docker_config_with_fallback(settings: Settings) -> "DockerConfig":
    """Get Docker configuration with CLI context integration.

    Args:
        settings: Main application settings

    Returns:
        Docker configuration with CLI overrides applied
    """
    try:
        from ccproxy.plugins.docker.config import DockerConfig

        # Create base config
        config = DockerConfig()

        # Apply CLI context overrides if available
        cli_context = settings.get_cli_context()
        if cli_context:
            # Apply CLI overrides to config
            if cli_context.get("docker_image"):
                config.docker_image = cli_context["docker_image"]

            if cli_context.get("docker_home"):
                config.docker_home_directory = cli_context["docker_home"]

            if cli_context.get("docker_workspace"):
                config.docker_workspace_directory = cli_context["docker_workspace"]

            if cli_context.get("docker_env"):
                config.docker_environment.extend(cli_context["docker_env"])

            if cli_context.get("docker_volume"):
                config.docker_volumes.extend(cli_context["docker_volume"])

            if cli_context.get("user_mapping_enabled") is not None:
                config.user_mapping_enabled = cli_context["user_mapping_enabled"]

            if cli_context.get("user_uid"):
                config.user_uid = cli_context["user_uid"]

            if cli_context.get("user_gid"):
                config.user_gid = cli_context["user_gid"]

        return config

    except ImportError as e:
        raise RuntimeError(
            f"Docker plugin not available: {e}. Docker functionality is not installed."
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
