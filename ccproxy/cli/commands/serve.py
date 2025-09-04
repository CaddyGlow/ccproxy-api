"""Serve command for CCProxy API server - consolidates server-related commands."""

import os
import subprocess
from pathlib import Path
from typing import Annotated, Any

import typer
import uvicorn
from click import get_current_context
from structlog import get_logger

from ccproxy.cli.helpers import (
    get_rich_toolkit,
    is_running_in_docker,
    warning,
)
from ccproxy.config.settings import (
    ConfigurationError,
    Settings,
)
from ccproxy.core._version import __version__
from ccproxy.docker import (
    create_docker_adapter,
)

from ..options.security_options import validate_auth_token
from ..options.server_options import (
    validate_log_level,
    validate_port,
)


def get_config_path_from_context() -> Path | None:
    """Get config path from typer context if available."""
    try:
        ctx = get_current_context()
        if ctx and ctx.obj and "config_path" in ctx.obj:
            config_path = ctx.obj["config_path"]
            return config_path if config_path is None else Path(config_path)
    except RuntimeError:
        pass
    return None


def _show_api_usage_info(toolkit: Any, settings: Settings) -> None:
    """Show API usage information when auth token is configured."""
    from rich.console import Console
    from rich.syntax import Syntax

    toolkit.print_title("API Client Configuration", tag="config")

    anthropic_base_url = f"http://{settings.server.host}:{settings.server.port}"
    openai_base_url = f"http://{settings.server.host}:{settings.server.port}/openai"

    toolkit.print("Environment Variables for API Clients:", tag="info")
    toolkit.print_line()

    console = Console()

    auth_token = "YOUR_AUTH_TOKEN" if settings.security.auth_token else "NOT_SET"
    exports = f"""export ANTHROPIC_API_KEY={auth_token}
export ANTHROPIC_BASE_URL={anthropic_base_url}
export OPENAI_API_KEY={auth_token}
export OPENAI_BASE_URL={openai_base_url}"""

    console.print(Syntax(exports, "bash", theme="monokai", background_color="default"))
    toolkit.print_line()


def _run_docker_server(
    settings: Settings,
    docker_image: str | None = None,
    docker_env: list[str] | None = None,
    docker_volume: list[str] | None = None,
    docker_arg: list[str] | None = None,
    docker_home: str | None = None,
    docker_workspace: str | None = None,
    user_mapping_enabled: bool | None = None,
    user_uid: int | None = None,
    user_gid: int | None = None,
) -> None:
    """Run the server using Docker."""
    toolkit = get_rich_toolkit()
    logger = get_logger(__name__)

    docker_env = docker_env or []
    docker_volume = docker_volume or []
    docker_arg = docker_arg or []

    docker_env_dict = {}
    for env_var in docker_env:
        if "=" in env_var:
            key, value = env_var.split("=", 1)
            docker_env_dict[key] = value

    if settings.server.reload:
        docker_env_dict["RELOAD"] = "true"
    docker_env_dict["PORT"] = str(settings.server.port)
    docker_env_dict["HOST"] = "0.0.0.0"

    toolkit.print_line()

    toolkit.print_title("Docker Configuration Summary", tag="config")

    home_dir = docker_home or settings.docker.docker_home_directory
    workspace_dir = docker_workspace or settings.docker.docker_workspace_directory

    toolkit.print("Volumes:", tag="config")
    if home_dir:
        toolkit.print(f"  Home: {home_dir} → /data/home", tag="volume")
    if workspace_dir:
        toolkit.print(f"  Workspace: {workspace_dir} → /data/workspace", tag="volume")
    if docker_volume:
        for vol in docker_volume:
            toolkit.print(f"  Additional: {vol}", tag="volume")
    toolkit.print_line()

    toolkit.print("Environment Variables:", tag="config")
    key_env_vars = {
        "CLAUDE_HOME": "/data/home",
        "CLAUDE_WORKSPACE": "/data/workspace",
        "PORT": str(settings.server.port),
        "HOST": "0.0.0.0",
    }
    if settings.server.reload:
        key_env_vars["RELOAD"] = "true"

    for key, value in key_env_vars.items():
        toolkit.print(f"  {key}={value}", tag="env")

    for env_var in docker_env:
        toolkit.print(f"  {env_var}", tag="env")

    if settings.logging.level == "DEBUG":
        toolkit.print_line()
        toolkit.print_title("Debug: All Environment Variables", tag="debug")
        all_env = {**docker_env_dict}
        for key, value in sorted(all_env.items()):
            toolkit.print(f"  {key}={value}", tag="debug")

    toolkit.print_line()

    toolkit.print_line()

    if settings.security.auth_token:
        _show_api_usage_info(toolkit, settings)

    adapter = create_docker_adapter()
    image, volumes, environment, command, user_context, _ = (
        adapter.build_docker_run_args(
            settings,
            command=["ccproxy", "serve"],
            docker_image=docker_image,
            docker_env=[f"{k}={v}" for k, v in docker_env_dict.items()],
            docker_volume=docker_volume,
            docker_arg=docker_arg,
            docker_home=docker_home,
            docker_workspace=docker_workspace,
            user_mapping_enabled=user_mapping_enabled,
            user_uid=user_uid,
            user_gid=user_gid,
        )
    )

    logger.info(
        "docker_server_config",
        configured_image=settings.docker.docker_image,
        effective_image=image,
    )

    ports = [f"{settings.server.port}:{settings.server.port}"]

    adapter = create_docker_adapter()
    adapter.exec_container(
        image=image,
        volumes=volumes,
        environment=environment,
        command=command,
        user_context=user_context,
        ports=ports,
    )


def _run_local_server(settings: Settings) -> None:
    """Run the server locally."""
    in_docker = is_running_in_docker()
    toolkit = get_rich_toolkit()
    logger = get_logger(__name__)

    if in_docker:
        toolkit.print_title(
            f"Starting CCProxy API server in {warning('docker')}",
            tag="docker",
        )
        toolkit.print(
            f"uid={warning(str(os.getuid()))} gid={warning(str(os.getgid()))}"
        )
        toolkit.print(f"HOME={os.environ['HOME']}")

    if settings.security.auth_token:
        _show_api_usage_info(toolkit, settings)

    logger.debug(
        "server_starting",
        host=settings.server.host,
        port=settings.server.port,
        url=f"http://{settings.server.host}:{settings.server.port}",
    )

    reload_includes = None
    if settings.server.reload:
        reload_includes = ["ccproxy", "pyproject.toml", "uv.lock", "plugins"]

    # container = create_service_container(settings)

    uvicorn.run(
        # app=create_app(container),
        app="ccproxy.api.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=settings.server.workers,
        log_config=None,
        access_log=False,
        server_header=False,
        date_header=False,
        reload_includes=reload_includes,
    )


def api(
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            help="Path to configuration file (TOML, JSON, or YAML)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            rich_help_panel="Configuration",
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            "--port",
            "-p",
            help="Port to run the server on",
            callback=validate_port,
            rich_help_panel="Server Settings",
        ),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            "--host",
            "-h",
            help="Host to bind the server to",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    reload: Annotated[
        bool | None,
        typer.Option(
            "--reload/--no-reload",
            help="Enable auto-reload for development",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL). Use WARNING for minimal output.",
            callback=validate_log_level,
            rich_help_panel="Server Settings",
        ),
    ] = None,
    log_file: Annotated[
        str | None,
        typer.Option(
            "--log-file",
            help="Path to JSON log file. If specified, logs will be written to this file in JSON format",
            rich_help_panel="Server Settings",
        ),
    ] = None,
    auth_token: Annotated[
        str | None,
        typer.Option(
            "--auth-token",
            help="Bearer token for API authentication",
            callback=validate_auth_token,
            rich_help_panel="Security Settings",
        ),
    ] = None,
    plugin_setting: Annotated[
        list[str] | None,
        typer.Option(
            "--plugin-setting",
            help="Set a plugin setting, e.g., claude_sdk.cli_path=/path/to/claude",
            rich_help_panel="Plugin Settings",
        ),
    ] = None,
    docker: Annotated[
        bool,
        typer.Option(
            "--docker",
            "-d",
            help="Run API server using Docker instead of local execution",
        ),
    ] = False,
    docker_image: Annotated[
        str | None,
        typer.Option(
            "--docker-image",
            help="Docker image to use (overrides configuration)",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_env: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-env",
            "-e",
            help="Environment variables to pass to Docker container",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_volume: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-volume",
            "-v",
            help="Volume mounts for Docker container",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-arg",
            help="Additional arguments to pass to docker run",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_home: Annotated[
        str | None,
        typer.Option(
            "--docker-home",
            help="Override the home directory for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_workspace: Annotated[
        str | None,
        typer.Option(
            "--docker-workspace",
            help="Override the workspace directory for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_mapping_enabled: Annotated[
        bool | None,
        typer.Option(
            "--user-mapping/--no-user-mapping",
            help="Enable user mapping for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_uid: Annotated[
        int | None,
        typer.Option(
            "--user-uid",
            help="User UID for Docker user mapping",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_gid: Annotated[
        int | None,
        typer.Option(
            "--user-gid",
            help="User GID for Docker user mapping",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    no_network_calls: Annotated[
        bool,
        typer.Option(
            "--no-network-calls",
            help="Disable all network calls (version checks and pricing updates)",
            rich_help_panel="Privacy Settings",
        ),
    ] = False,
    disable_version_check: Annotated[
        bool,
        typer.Option(
            "--disable-version-check",
            help="Disable version update checks (prevents calls to GitHub API)",
            rich_help_panel="Privacy Settings",
        ),
    ] = False,
    disable_pricing_updates: Annotated[
        bool,
        typer.Option(
            "--disable-pricing-updates",
            help="Disable pricing data updates (prevents downloads from GitHub)",
            rich_help_panel="Privacy Settings",
        ),
    ] = False,
) -> None:
    """Start the CCProxy API server."""
    try:
        if config is None:
            config = get_config_path_from_context()

        scheduler_overrides = {}
        if no_network_calls:
            scheduler_overrides["pricing_update_enabled"] = False
            scheduler_overrides["version_check_enabled"] = False
        else:
            if disable_pricing_updates:
                scheduler_overrides["pricing_update_enabled"] = False
            if disable_version_check:
                scheduler_overrides["version_check_enabled"] = False

        settings = Settings.from_config(config_path=config)

        from ccproxy.core.logging import setup_logging

        setup_logging(
            json_logs=settings.logging.format == "json",
            log_level_name=settings.logging.level,
            log_file=settings.logging.file,
        )

        logger = get_logger(__name__)

        logger.debug(
            "configuration_loaded",
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.logging.level,
            log_file=settings.logging.file,
            docker_mode=docker,
            docker_image=settings.docker.docker_image if docker else None,
            auth_enabled=bool(settings.security.auth_token),
            duckdb_enabled=bool(
                (settings.plugins.get("duckdb_storage") or {}).get("enabled", False)
            ),
        )

        if docker:
            _run_docker_server(
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
            )
        else:
            _run_local_server(settings)

    except ConfigurationError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except OSError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(
            f"Server startup failed (port/permission issue): {e}", tag="error"
        )
        raise typer.Exit(1) from e
    except ImportError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Import error during server startup: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Error starting server: {e}", tag="error")
        raise typer.Exit(1) from e


def claude(
    args: Annotated[
        list[str] | None,
        typer.Argument(
            help="Arguments to pass to claude CLI (e.g. --version, doctor, config)",
        ),
    ] = None,
    docker: Annotated[
        bool,
        typer.Option(
            "--docker",
            "-d",
            help="Run claude command from docker image instead of local CLI",
        ),
    ] = False,
    docker_image: Annotated[
        str | None,
        typer.Option(
            "--docker-image",
            help="Docker image to use (overrides configuration)",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_env: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-env",
            "-e",
            help="Environment variables to pass to Docker container",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_volume: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-volume",
            "-v",
            help="Volume mounts for Docker container",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_arg: Annotated[
        list[str] | None,
        typer.Option(
            "--docker-arg",
            help="Additional arguments to pass to docker run",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_home: Annotated[
        str | None,
        typer.Option(
            "--docker-home",
            help="Override the home directory for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    docker_workspace: Annotated[
        str | None,
        typer.Option(
            "--docker-workspace",
            help="Override the workspace directory for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_mapping_enabled: Annotated[
        bool | None,
        typer.Option(
            "--user-mapping/--no-user-mapping",
            help="Enable user mapping for Docker",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_uid: Annotated[
        int | None,
        typer.Option(
            "--user-uid",
            help="User UID for Docker user mapping",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
    user_gid: Annotated[
        int | None,
        typer.Option(
            "--user-gid",
            help="User GID for Docker user mapping",
            rich_help_panel="Docker Settings",
        ),
    ] = None,
) -> None:
    """Execute claude CLI commands directly."""
    if args is None:
        args = []

    toolkit = get_rich_toolkit()

    try:
        logger = get_logger(__name__)
        logger.info(
            "cli_command_starting",
            command="claude",
            version=__version__,
            docker=docker,
            args=args if args else [],
        )

        settings = Settings.from_config(get_config_path_from_context())

        if docker:
            adapter = create_docker_adapter()
            toolkit.print_title(f"image {settings.docker.docker_image}", tag="docker")
            image, volumes, environment, command, user_context, _ = (
                adapter.build_docker_run_args(
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
                    command=["claude"] + (args or []),
                )
            )

            cmd_str = " ".join(command or [])
            logger.info(
                "docker_execution",
                image=image,
                command=" ".join(command or []),
                volumes_count=len(volumes),
                env_vars_count=len(environment),
            )
            toolkit.print(f"Executing: docker run ... {image} {cmd_str}", tag="docker")
            toolkit.print_line()

            adapter.exec_container(
                image=image,
                volumes=volumes,
                environment=environment,
                command=command,
                user_context=user_context,
            )
        else:
            import shutil
            from pathlib import Path

            from ccproxy.utils.binary_resolver import BinaryResolver

            claude_paths = [
                shutil.which("claude"),
                Path.home() / ".cache" / ".bun" / "bin" / "claude",
                Path.home() / ".local" / "bin" / "claude",
                Path("/usr/local/bin/claude"),
            ]

            claude_cmd: str | list[str] | None = None
            for path in claude_paths:
                if path and Path(str(path)).exists():
                    claude_cmd = str(path)
                    break

            if not claude_cmd:
                resolver = BinaryResolver()
                result = resolver.find_binary("claude", "@anthropic-ai/claude-code")
                if result:
                    claude_cmd = (
                        result.command[0] if result.is_direct else result.command
                    )

            if not claude_cmd:
                toolkit.print("Error: Claude CLI not found.", tag="error")
                toolkit.print(
                    "Please install Claude CLI.",
                    tag="error",
                )
                raise typer.Exit(1)

            if isinstance(claude_cmd, str):
                if not Path(claude_cmd).is_absolute():
                    claude_cmd = str(Path(claude_cmd).resolve())

                logger.info("local_claude_execution", claude_path=claude_cmd, args=args)
                toolkit.print(f"Executing: {claude_cmd} {' '.join(args)}", tag="claude")
                toolkit.print_line()

                try:
                    os.execvp(claude_cmd, [claude_cmd] + args)
                except OSError as e:
                    toolkit.print(f"Failed to execute command: {e}", tag="error")
                    raise typer.Exit(1) from e
            else:
                if not isinstance(claude_cmd, list):
                    raise ValueError("Expected list for package manager command")
                full_cmd = claude_cmd + args
                logger.info(
                    "local_claude_execution_via_package_manager",
                    command=full_cmd,
                    package_manager=claude_cmd[0],
                )
                toolkit.print(f"Executing: {' '.join(full_cmd)}", tag="claude")
                toolkit.print_line()

                try:
                    proc_result = subprocess.run(full_cmd, check=False)
                    raise typer.Exit(proc_result.returncode)
                except subprocess.SubprocessError as e:
                    toolkit.print(f"Failed to execute command: {e}", tag="error")
                    raise typer.Exit(1) from e

    except ConfigurationError as e:
        logger = get_logger(__name__)
        logger.error("cli_configuration_error", error=str(e), command="claude")
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except FileNotFoundError as e:
        logger = get_logger(__name__)
        logger.error("cli_command_not_found", error=str(e), command="claude")
        toolkit.print(f"Claude command not found: {e}", tag="error")
        raise typer.Exit(1) from e
    except OSError as e:
        logger = get_logger(__name__)
        logger.error("cli_os_error", error=str(e), command="claude")
        toolkit.print(f"System error executing claude command: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        logger = get_logger(__name__)
        logger.error("cli_unexpected_error", error=str(e), command="claude")
        toolkit.print(f"Error executing claude command: {e}", tag="error")
        raise typer.Exit(1) from e
