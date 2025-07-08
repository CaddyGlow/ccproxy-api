"""Claude command for executing Claude CLI commands directly."""

import logging
import os
from pathlib import Path

import typer
from click import get_current_context

# Removed duplicate import - _create_docker_adapter_from_settings is imported below
from ccproxy.cli.docker.adapter_factory import (
    _create_docker_adapter_from_settings,
)
from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
    config_manager,
)
from ccproxy.docker import (
    DockerEnv,
    DockerPath,
    DockerUserContext,
    DockerVolume,
    create_docker_adapter,
)
from ccproxy.utils.cli import get_rich_toolkit
from ccproxy.utils.logging import get_logger

from ..docker.params import (
    docker_arg_option,
    docker_env_option,
    docker_home_option,
    docker_image_option,
    docker_volume_option,
    docker_workspace_option,
    user_gid_option,
    user_mapping_option,
    user_uid_option,
)


# Logger will be configured by configuration manager
logger = get_logger(__name__)


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
        # No active click context (e.g., in tests)
        pass
    return None


# This function has been moved to claude_code_proxy.cli.docker.adapter_factory
# Import it from there instead (imported at top of file)


def claude(
    args: list[str] | None = typer.Argument(
        default=None,
        help="Arguments to pass to claude CLI (e.g. --version, doctor, config)",
    ),
    docker: bool = typer.Option(
        False,
        "--docker",
        "-d",
        help="Run claude command from docker image instead of local CLI",
    ),
    # Docker settings using shared parameters
    docker_image: str | None = docker_image_option(),
    docker_env: list[str] = docker_env_option(),
    docker_volume: list[str] = docker_volume_option(),
    docker_arg: list[str] = docker_arg_option(),
    docker_home: str | None = docker_home_option(),
    docker_workspace: str | None = docker_workspace_option(),
    user_mapping_enabled: bool | None = user_mapping_option(),
    user_uid: int | None = user_uid_option(),
    user_gid: int | None = user_gid_option(),
) -> None:
    """
    Execute claude CLI commands directly.

    This is a simple pass-through to the claude CLI executable
    found by the settings system or run from docker image.

    Examples:
        ccproxy claude -- --version
        ccproxy claude -- doctor
        ccproxy claude -- config
        ccproxy claude --docker -- --version
        ccproxy claude --docker --docker-image custom:latest -- --version
        ccproxy claude --docker --docker-env API_KEY=sk-... --docker-volume ./data:/data -- chat
    """
    # Handle None args case
    if args is None:
        args = []

    toolkit = get_rich_toolkit()

    try:
        # Load settings using configuration manager
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )

        if docker:
            # Prepare Docker execution using new adapter

            toolkit.print_title(
                f"image {settings.docker_settings.docker_image}", tag="docker"
            )
            image, volumes, environment, command, user_context, additional_args = (
                _create_docker_adapter_from_settings(
                    settings,
                    docker_image=docker_image,
                    docker_env=docker_env,
                    docker_volume=docker_volume,
                    docker_arg=docker_arg,
                    docker_home=docker_home,
                    docker_workspace=docker_workspace,
                    user_mapping_enabled=user_mapping_enabled,
                    user_uid=user_uid,
                    user_gid=user_gid,
                    command=["claude"],
                    cmd_args=args,
                )
            )

            cmd_str = " ".join(command or [])
            toolkit.print(f"Executing: docker run ... {image} {cmd_str}", tag="docker")
            toolkit.print_line()

            # Execute using the new Docker adapter
            adapter = create_docker_adapter()
            adapter.exec_container(
                image=image,
                volumes=volumes,
                environment=environment,
                command=command,
                user_context=user_context,
            )
        else:
            # Get claude path from settings
            claude_path = settings.claude_cli_path
            if not claude_path:
                toolkit.print("Error: Claude CLI not found.", tag="error")
                toolkit.print(
                    "Please install Claude CLI or configure claude_cli_path.",
                    tag="error",
                )
                raise typer.Exit(1)

            # Resolve to absolute path
            if not Path(claude_path).is_absolute():
                claude_path = str(Path(claude_path).resolve())

            toolkit.print(f"Executing: {claude_path} {' '.join(args)}", tag="claude")
            toolkit.print_line()

            # Execute command directly
            try:
                # Use os.execvp to replace current process with claude
                # This hands over full control to claude, including signal handling
                os.execvp(claude_path, [claude_path] + args)
            except OSError as e:
                toolkit.print(f"Failed to execute command: {e}", tag="error")
                raise typer.Exit(1) from e

    except ConfigurationError as e:
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit.print(f"Error executing claude command: {e}", tag="error")
        raise typer.Exit(1) from e
