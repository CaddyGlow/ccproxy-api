"""Docker plugin with CLI extensions."""

from typing import Any

import ccproxy.core.logging
from ccproxy.core.plugins import (
    BaseProviderPluginFactory,
    PluginContext,
    PluginManifest,
    ProviderPluginRuntime,
)
from ccproxy.core.plugins.declaration import CliArgumentSpec

from .adapter import DockerAdapter
from .config import DockerConfig


logger = ccproxy.core.logging.get_plugin_logger(__name__)


class DockerRuntime(ProviderPluginRuntime):
    """Runtime for Docker plugin."""

    def __init__(self, manifest: PluginManifest):
        """Initialize runtime."""
        super().__init__(manifest)

    async def _on_initialize(self) -> None:
        """Initialize the Docker plugin."""
        await super()._on_initialize()

        if not self.context:
            raise RuntimeError("Context not set")

        # Get CLI arguments from context
        settings = self.context.get("settings")
        if settings:
            cli_context = settings.get_cli_context()

            # Process Docker CLI flags and update config
            config = self.context.get("config")
            if config and isinstance(config, DockerConfig):
                self._apply_cli_overrides(cli_context, config)

        config = self.context.get("config")
        docker_image = (
            config.docker_image if config and isinstance(config, DockerConfig) else None
        )

        logger.debug(
            "plugin_initialized",
            plugin="docker",
            version="0.1.0",
            status="initialized",
            docker_image=docker_image,
        )

    def _apply_cli_overrides(
        self, cli_context: dict[str, Any], config: DockerConfig
    ) -> None:
        """Apply CLI flag overrides to Docker config."""
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

        logger.debug("docker_cli_overrides_applied", cli_overrides=cli_context)


class DockerFactory(BaseProviderPluginFactory):
    """Factory for Docker plugin."""

    # Plugin configuration via class attributes
    plugin_name = "docker"
    plugin_description = "Docker container management for CCProxy"
    runtime_class = DockerRuntime
    adapter_class = DockerAdapter
    config_class = DockerConfig

    # CLI extension declarations - all Docker-related CLI arguments
    cli_arguments = [
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker",
            argument_type=bool,
            help_text="Run using Docker instead of local execution",
            default=False,
            typer_kwargs={
                "is_flag": True,
                "flag_value": True,
                "option": ["--docker", "-d"],
            },
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_image",
            argument_type=str,
            help_text="Docker image to use (overrides configuration)",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_env",
            argument_type=list[str],
            help_text="Environment variables to pass to Docker container",
            typer_kwargs={
                "rich_help_panel": "Docker Settings",
                "option": ["--docker-env", "-e"],
            },
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_volume",
            argument_type=list[str],
            help_text="Volume mounts for Docker container",
            typer_kwargs={
                "rich_help_panel": "Docker Settings",
                "option": ["--docker-volume", "-v"],
            },
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_arg",
            argument_type=list[str],
            help_text="Additional arguments to pass to docker run",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_home",
            argument_type=str,
            help_text="Override the home directory for Docker",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="docker_workspace",
            argument_type=str,
            help_text="Override the workspace directory for Docker",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="user_mapping_enabled",
            argument_type=bool,
            help_text="Enable user mapping for Docker",
            typer_kwargs={
                "rich_help_panel": "Docker Settings",
                "option": ["--user-mapping/--no-user-mapping"],
            },
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="user_uid",
            argument_type=int,
            help_text="User UID for Docker user mapping",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
        CliArgumentSpec(
            target_command="serve",
            argument_name="user_gid",
            argument_type=int,
            help_text="User GID for Docker user mapping",
            typer_kwargs={"rich_help_panel": "Docker Settings"},
        ),
    ]

    async def create_adapter(self, context: PluginContext) -> DockerAdapter:
        """Create Docker adapter instance."""
        config = context.get("config")
        if not isinstance(config, DockerConfig):
            config = DockerConfig()

        return DockerAdapter(config=config)


# Export factory instance
factory = DockerFactory()
