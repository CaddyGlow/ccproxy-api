"""Serve command for CCProxy API server - consolidates server-related commands."""

import json
import os
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
    config_manager,
)
from ccproxy.core.async_utils import get_root_package_name
from ccproxy.docker import (
    create_docker_adapter,
)

from ..docker import (
    _create_docker_adapter_from_settings,
)
from ..options.claude_options import (
    ClaudeOptions,
    validate_claude_cli_path,
    validate_cwd,
    validate_max_thinking_tokens,
    validate_max_turns,
    validate_permission_mode,
    validate_pool_size,
    validate_sdk_message_mode,
    validate_system_prompt_injection_mode,
)
from ..options.security_options import SecurityOptions, validate_auth_token
from ..options.server_options import (
    ServerOptions,
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
        # No active click context (e.g., in tests)
        pass
    return None


def _show_api_usage_info(toolkit: Any, settings: Settings) -> None:
    """Show API usage information when auth token is configured."""
    from rich.console import Console
    from rich.syntax import Syntax

    toolkit.print_title("API Client Configuration", tag="config")

    # Determine the base URLs
    anthropic_base_url = f"http://{settings.server.host}:{settings.server.port}"
    openai_base_url = f"http://{settings.server.host}:{settings.server.port}/openai"

    # Show environment variable exports using code blocks
    toolkit.print("Environment Variables for API Clients:", tag="info")
    toolkit.print_line()

    # Use rich console for code blocks
    console = Console()

    exports = f"""export ANTHROPIC_API_KEY={settings.security.auth_token}
export ANTHROPIC_BASE_URL={anthropic_base_url}
export OPENAI_API_KEY={settings.security.auth_token}
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

    # Add server configuration to Docker environment
    if settings.server.reload:
        docker_env_dict["RELOAD"] = "true"
    docker_env_dict["PORT"] = str(settings.server.port)
    docker_env_dict["HOST"] = "0.0.0.0"

    # Display startup information
    # toolkit.print_title(
    #     "Starting CCProxy API server with Docker", tag="docker"
    # )
    # toolkit.print(
    #     f"Server will be available at: http://{settings.server.host}:{settings.server.port}",
    #     tag="info",
    # )
    toolkit.print_line()

    # Show Docker configuration summary
    toolkit.print_title("Docker Configuration Summary", tag="config")

    # Determine effective directories for volume mapping
    home_dir = docker_home or settings.docker.docker_home_directory
    workspace_dir = docker_workspace or settings.docker.docker_workspace_directory

    # Show volume information
    toolkit.print("Volumes:", tag="config")
    if home_dir:
        toolkit.print(f"  Home: {home_dir} → /data/home", tag="volume")
    if workspace_dir:
        toolkit.print(f"  Workspace: {workspace_dir} → /data/workspace", tag="volume")
    if docker_volume:
        for vol in docker_volume:
            toolkit.print(f"  Additional: {vol}", tag="volume")
    toolkit.print_line()

    # Show environment information
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

    # Show additional environment variables from CLI
    for env_var in docker_env:
        toolkit.print(f"  {env_var}", tag="env")

    # Show debug environment information if log level is DEBUG
    if settings.server.log_level == "DEBUG":
        toolkit.print_line()
        toolkit.print_title("Debug: All Environment Variables", tag="debug")
        all_env = {**docker_env_dict}
        for key, value in sorted(all_env.items()):
            toolkit.print(f"  {key}={value}", tag="debug")

    toolkit.print_line()

    toolkit.print_line()

    # Show API usage information if auth token is configured
    if settings.security.auth_token:
        _show_api_usage_info(toolkit, settings)

    # Execute using the new Docker adapter
    image, volumes, environment, command, user_context, additional_args = (
        _create_docker_adapter_from_settings(
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

    # Add port mapping
    ports = [f"{settings.server.port}:{settings.server.port}"]

    # Create Docker adapter and execute
    adapter = create_docker_adapter()
    adapter.exec_container(
        image=image,
        volumes=volumes,
        environment=environment,
        command=command,
        user_context=user_context,
        ports=ports,
    )


def _run_local_server(settings: Settings, cli_overrides: dict[str, Any]) -> None:
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
    # else:
    #     toolkit.print_title("Starting CCProxy API server", tag="local")

    # toolkit.print(
    #     f"Server will be available at: http://{settings.server.host}:{settings.server.port}",
    #     tag="info",
    # )

    # toolkit.print_line()

    # Show API usage information if auth token is configured
    if settings.security.auth_token:
        _show_api_usage_info(toolkit, settings)

    # Set environment variables for server to access CLI overrides
    if cli_overrides:
        os.environ["CCPROXY_CONFIG_OVERRIDES"] = json.dumps(cli_overrides)

    logger.debug(
        "server_starting",
        host=settings.server.host,
        port=settings.server.port,
        url=f"http://{settings.server.host}:{settings.server.port}",
    )

    reload_includes = None
    if settings.server.reload:
        reload_includes = ["ccproxy", "pyproject.toml", "uv.lock"]

    # Run uvicorn with our already configured logging
    uvicorn.run(
        app=f"{get_root_package_name()}.api.app:create_app",
        factory=True,
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=None,  # ,settings.workers,
        log_config=None,
        access_log=False,  # Disable uvicorn's default access logs
        server_header=False,  # Disable uvicorn's server header to preserve upstream headers
        reload_includes=reload_includes,
        # log_config=get_uvicorn_log_config(),
    )


def api(
    # Configuration
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
    # Server options
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
    use_terminal_permission_handler: Annotated[
        bool,
        typer.Option(
            "--terminal-permission-handler",
            help="Enable terminal permission terminal handler",
            rich_help_panel="Server Settings",
        ),
    ] = False,
    # Security options
    auth_token: Annotated[
        str | None,
        typer.Option(
            "--auth-token",
            help="Bearer token for API authentication",
            callback=validate_auth_token,
            rich_help_panel="Security Settings",
        ),
    ] = None,
    # Claude options
    max_thinking_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-thinking-tokens",
            help="Maximum thinking tokens for Claude Code",
            callback=validate_max_thinking_tokens,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    allowed_tools: Annotated[
        str | None,
        typer.Option(
            "--allowed-tools",
            help="List of allowed tools (comma-separated)",
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    disallowed_tools: Annotated[
        str | None,
        typer.Option(
            "--disallowed-tools",
            help="List of disallowed tools (comma-separated)",
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    claude_cli_path: Annotated[
        str | None,
        typer.Option(
            "--claude-cli-path",
            help="Path to Claude CLI executable",
            callback=validate_claude_cli_path,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    append_system_prompt: Annotated[
        str | None,
        typer.Option(
            "--append-system-prompt",
            help="Additional system prompt to append",
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    permission_mode: Annotated[
        str | None,
        typer.Option(
            "--permission-mode",
            help="Permission mode: default, acceptEdits, or bypassPermissions",
            callback=validate_permission_mode,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    max_turns: Annotated[
        int | None,
        typer.Option(
            "--max-turns",
            help="Maximum conversation turns",
            callback=validate_max_turns,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    cwd: Annotated[
        str | None,
        typer.Option(
            "--cwd",
            help="Working directory path",
            callback=validate_cwd,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    permission_prompt_tool_name: Annotated[
        str | None,
        typer.Option(
            "--permission-prompt-tool-name",
            help="Permission prompt tool name",
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    sdk_message_mode: Annotated[
        str | None,
        typer.Option(
            "--sdk-message-mode",
            help="SDK message handling mode: forward (direct SDK blocks), ignore (skip blocks), formatted (XML tags with JSON data)",
            callback=validate_sdk_message_mode,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    sdk_enable_pool: Annotated[
        bool,
        typer.Option(
            "--sdk-enable-pool",
            help="Enable Claude SDK client connection pooling for improved performance",
            rich_help_panel="Claude Settings",
        ),
    ] = False,
    sdk_pool_size: Annotated[
        int | None,
        typer.Option(
            "--sdk-pool-size",
            help="Number of clients to maintain in the pool (1-20)",
            callback=validate_pool_size,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    system_prompt_injection_mode: Annotated[
        str | None,
        typer.Option(
            "--system-prompt-injection-mode",
            help="System prompt injection mode: minimal (Claude Code ID only), full (all detected system messages)",
            callback=validate_system_prompt_injection_mode,
            rich_help_panel="Claude Settings",
        ),
    ] = None,
    builtin_permissions: Annotated[
        bool,
        typer.Option(
            "--builtin-permissions/--no-builtin-permissions",
            help="Enable built-in permission handling infrastructure (MCP server and SSE endpoints). When disabled, users can configure custom MCP servers and permission tools.",
            rich_help_panel="Claude Settings",
        ),
    ] = True,
    # Core settings
    docker: Annotated[
        bool,
        typer.Option(
            "--docker",
            "-d",
            help="Run API server using Docker instead of local execution",
        ),
    ] = False,
    # Docker settings using shared parameters
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
    """
    Start the CCProxy API server.

    This command starts the API server either locally or in Docker.
    The server provides both Anthropic and OpenAI-compatible endpoints.

    All configuration options can be provided via CLI parameters,
    which override values from configuration files and environment variables.

    Examples:
        ccproxy serve
        ccproxy serve --port 8080 --reload
        ccproxy serve --docker
        ccproxy serve --docker --docker-image custom:latest --port 8080
        ccproxy serve --max-thinking-tokens 10000 --allowed-tools Read,Write,Bash
        ccproxy serve --port 8080 --workers 4
    """
    try:
        # Early logging - use basic print until logging is configured
        # We'll log this properly after logging is configured

        # Get config path from context if not provided directly
        if config is None:
            config = get_config_path_from_context()

        # Create option containers for better organization
        server_options = ServerOptions(
            port=port,
            host=host,
            reload=reload,
            log_level=log_level,
            log_file=log_file,
            use_terminal_confirmation_handler=use_terminal_permission_handler,
        )

        claude_options = ClaudeOptions(
            max_thinking_tokens=max_thinking_tokens,
            allowed_tools=allowed_tools,
            disallowed_tools=disallowed_tools,
            claude_cli_path=claude_cli_path,
            append_system_prompt=append_system_prompt,
            permission_mode=permission_mode,
            max_turns=max_turns,
            cwd=cwd,
            permission_prompt_tool_name=permission_prompt_tool_name,
            sdk_message_mode=sdk_message_mode,
            sdk_enable_pool=sdk_enable_pool,
            sdk_pool_size=sdk_pool_size,
            system_prompt_injection_mode=system_prompt_injection_mode,
            builtin_permissions=builtin_permissions,
        )

        security_options = SecurityOptions(auth_token=auth_token)

        # Extract CLI overrides from structured option containers
        cli_overrides = config_manager.get_cli_overrides_from_args(
            # Server options
            host=server_options.host,
            port=server_options.port,
            reload=server_options.reload,
            log_level=server_options.log_level,
            log_file=server_options.log_file,
            use_terminal_confirmation_handler=server_options.use_terminal_confirmation_handler,
            # Security options
            auth_token=security_options.auth_token,
            # Claude options
            claude_cli_path=claude_options.claude_cli_path,
            max_thinking_tokens=claude_options.max_thinking_tokens,
            allowed_tools=claude_options.allowed_tools,
            disallowed_tools=claude_options.disallowed_tools,
            append_system_prompt=claude_options.append_system_prompt,
            permission_mode=claude_options.permission_mode,
            max_turns=claude_options.max_turns,
            permission_prompt_tool_name=claude_options.permission_prompt_tool_name,
            cwd=claude_options.cwd,
            sdk_message_mode=claude_options.sdk_message_mode,
            sdk_enable_pool=claude_options.sdk_enable_pool,
            sdk_pool_size=claude_options.sdk_pool_size,
            system_prompt_injection_mode=claude_options.system_prompt_injection_mode,
            builtin_permissions=claude_options.builtin_permissions,
        )

        # Load settings with CLI overrides
        settings = config_manager.load_settings(
            config_path=config, cli_overrides=cli_overrides
        )

        # Set up logging once with the effective log level
        # Import here to avoid circular import

        from ccproxy.core.logging import setup_logging

        # Always reconfigure logging to ensure log level changes are picked up
        # Use JSON logs if explicitly requested via env var
        print(f"{settings.server.log_level} {settings.server.log_file}")
        setup_logging(
            json_logs=settings.server.log_format == "json",
            log_level_name=settings.server.log_level,
            log_file=settings.server.log_file,
        )

        # Re-get logger after logging is configured
        logger = get_logger(__name__)

        # Test debug logging
        logger.debug(
            "Debug logging is enabled",
            effective_log_level=server_options.log_level or settings.server.log_level,
        )

        # Log CLI command that was deferred
        logger.info(
            "cli_command_starting",
            command="serve",
            docker=docker,
            port=server_options.port,
            host=server_options.host,
            config_path=str(config) if config else None,
        )

        # Log effective configuration
        logger.debug(
            "configuration_loaded",
            host=settings.server.host,
            port=settings.server.port,
            log_level=settings.server.log_level,
            log_file=settings.server.log_file,
            docker_mode=docker,
            docker_image=settings.docker.docker_image if docker else None,
            auth_enabled=bool(settings.security.auth_token),
            duckdb_enabled=settings.observability.duckdb_enabled,
            duckdb_path=settings.observability.duckdb_path
            if settings.observability.duckdb_enabled
            else None,
            claude_cli_path=settings.claude.cli_path,
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
            _run_local_server(settings, cli_overrides)

    except ConfigurationError as e:
        toolkit = get_rich_toolkit()
        toolkit.print(f"Configuration error: {e}", tag="error")
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
    # Docker settings using shared parameters
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
        # Logger will be configured by configuration manager
        logger = get_logger(__name__)
        # Log CLI command execution start
        logger.info(
            "cli_command_starting",
            command="claude",
            docker=docker,
            args=args if args else [],
        )

        # Load settings using configuration manager
        settings = config_manager.load_settings(
            config_path=get_config_path_from_context()
        )

        if docker:
            # Prepare Docker execution using new adapter

            toolkit.print_title(f"image {settings.docker.docker_image}", tag="docker")
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
            logger.info(
                "docker_execution",
                image=image,
                command=" ".join(command or []),
                volumes_count=len(volumes),
                env_vars_count=len(environment),
            )
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
            claude_path = settings.claude.cli_path
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

            logger.info("local_claude_execution", claude_path=claude_path, args=args)
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
        logger.error("cli_configuration_error", error=str(e), command="claude")
        toolkit.print(f"Configuration error: {e}", tag="error")
        raise typer.Exit(1) from e
    except Exception as e:
        logger.error("cli_unexpected_error", error=str(e), command="claude")
        toolkit.print(f"Error executing claude command: {e}", tag="error")
        raise typer.Exit(1) from e
