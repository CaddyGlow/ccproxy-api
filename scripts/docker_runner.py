#!/usr/bin/env python
"""Helper CLI for building and running the ccproxy Docker image.

This script uses the internal Docker plugin to mirror the legacy `--docker`
experience without restoring that flag on the main CLI. It assembles sensible
volume mounts so the container can reuse your local credentials and config.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer

from ccproxy.plugins.docker.adapter import DockerAdapter
from ccproxy.plugins.docker.config import DockerConfig
from ccproxy.plugins.docker.models import DockerUserContext
from ccproxy.plugins.docker.stream_process import (
    OutputMiddleware,
    RawPassthroughMiddleware,
)


APP = typer.Typer(help="Build and run the ccproxy Docker image with standard mounts")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = "ccproxy:local"
DEFAULT_PORT = 8000
DEFAULT_CONTEXT = PROJECT_ROOT
DEFAULT_WORKSPACE_MOUNT = "/workspace"
CONTAINER_HOME = "/home/appuser"
CONTAINER_WORKSPACE = DEFAULT_WORKSPACE_MOUNT
CONTAINER_CONFIG = f"{CONTAINER_HOME}/.config:ro"
CONTAINER_CACHE = f"{CONTAINER_HOME}/.cache/ccproxy"
CONTAINER_CLAUDE = f"{CONTAINER_HOME}/.claude"
CONTAINER_CODEX = f"{CONTAINER_HOME}/.codex"
CONTAINER_GH = f"{CONTAINER_HOME}/.config/gh"
CONTAINER_INIT_SCRIPT = f"{CONTAINER_HOME}/.ccproxy-init.sh"


class StreamPrinter(OutputMiddleware[None]):
    """Print process output directly without buffering it."""

    async def process(self, line: str, stream_type: str) -> None:
        if stream_type == "stdout":
            typer.echo(line)
        else:
            typer.echo(line, err=True)
        return None


def _split_image_reference(image: str) -> tuple[str, str]:
    """Split an image reference into repository and tag."""
    if ":" in image and not image.endswith(":"):
        repo, candidate = image.rsplit(":", 1)
        if "/" in candidate:
            # Handles hosts with ports like localhost:5000/image
            return image, "latest"
        return repo, candidate or "latest"
    return image, "latest"


def _ensure_directory(path: Path) -> Path:
    """Ensure a directory exists and return its resolved path."""
    expanded = path.expanduser()
    expanded.mkdir(parents=True, exist_ok=True)
    return expanded.resolve()


def _parse_env(env_pairs: Iterable[str]) -> dict[str, str]:
    """Convert KEY=VALUE strings into a dictionary."""
    parsed: dict[str, str] = {}
    for pair in env_pairs:
        if "=" not in pair:
            raise typer.BadParameter(f"Invalid env assignment '{pair}'. Use KEY=VALUE.")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Environment variable name missing in '{pair}'.")
        parsed[key] = value
    return parsed


def _parse_volumes(volume_specs: Iterable[str]) -> list[tuple[str, str]]:
    """Convert HOST:CONTAINER specs into volume tuples."""
    volumes: list[tuple[str, str]] = []
    for spec in volume_specs:
        if ":" not in spec:
            raise typer.BadParameter(
                f"Invalid volume specification '{spec}'. Use host:container"
            )
        host, container = spec.split(":", 1)
        host_path = _ensure_directory(Path(host))
        if not container:
            raise typer.BadParameter(f"Container path missing in volume '{spec}'")
        volumes.append((str(host_path), container))
    return volumes


def _parse_apt_packages(values: Iterable[str]) -> list[str]:
    """Normalize apt package arguments, splitting on commas."""

    raw_tokens: list[str] = []
    for raw in values:
        for token in raw.split(","):
            trimmed = token.strip()
            if trimmed:
                raw_tokens.append(trimmed)

    seen: set[str] = set()
    packages: list[str] = []
    for pkg in raw_tokens:
        if pkg not in seen:
            seen.add(pkg)
            packages.append(pkg)
    return packages


def _default_mounts(workspace: Path, workspace_mount: str) -> list[tuple[str, str]]:
    """Return the default host/container mounts used for Docker runs."""
    mounts: list[tuple[str, str]] = []

    def add(host: Path, container: str) -> None:
        mounts.append((str(_ensure_directory(host)), container))

    add(workspace, workspace_mount)
    add(Path.home() / ".codex", CONTAINER_CODEX)
    add(Path.home() / ".claude", CONTAINER_CLAUDE)
    # add(Path.home() / ".config/gh", CONTAINER_GH)
    # add(Path.home() / ".config/ccproxy", CONTAINER_CONFIG)
    add(Path.home() / ".config", CONTAINER_CONFIG)
    add(Path.home() / ".cache/ccproxy", CONTAINER_CACHE)

    return mounts


async def _build_image(
    adapter: DockerAdapter,
    image: str,
    context: Path,
    no_cache: bool,
    raw_output: bool = False,
    dockerfile: Path | None = None,
) -> int:
    repo, tag = _split_image_reference(image)
    middleware: OutputMiddleware[Any] | None
    if raw_output:
        middleware = RawPassthroughMiddleware()
    else:
        middleware = StreamPrinter()

    rc, _stdout, _stderr = await adapter.build_image(
        context,
        image_name=repo,
        image_tag=tag,
        no_cache=no_cache,
        dockerfile_path=dockerfile,
        middleware=middleware,
    )
    return rc


async def _ensure_image(
    adapter: DockerAdapter,
    image: str,
    context: Path,
    no_cache: bool,
    auto_build: bool,
    raw_output: bool = False,
    dockerfile: Path | None = None,
) -> None:
    repo, tag = _split_image_reference(image)
    if await adapter.image_exists(repo, tag):
        return
    if not auto_build:
        raise typer.BadParameter(
            f"Image '{image}' not found locally. Run 'docker_runner.py build' or remove --no-build."
        )
    dockerfile_note = f" using {dockerfile}" if dockerfile else ""
    typer.echo(
        f"Image '{image}' not found. Building from {context}{dockerfile_note}..."
    )
    rc = await _build_image(
        adapter,
        image,
        context,
        no_cache,
        raw_output=raw_output,
        dockerfile=dockerfile,
    )
    if rc != 0:
        raise typer.Exit(rc)


async def _run_container(
    adapter: DockerAdapter,
    image: str,
    command: list[str] | None,
    volumes: list[tuple[str, str]],
    environment: dict[str, str],
    ports: list[str],
    user_mapping: bool,
    extra_args: list[str] | None = None,
    raw_output: bool = False,
) -> int:
    user_context: DockerUserContext | None = None
    if user_mapping:
        try:
            user_context = DockerUserContext.detect_current_user()
            user_context.home_path = None
            user_context.workspace_path = None
        except Exception as exc:  # pragma: no cover - defensive path
            typer.echo(
                f"Warning: failed to detect user for mapping, running as root instead ({exc}).",
                err=True,
            )
            user_context = None

    middleware: OutputMiddleware[Any] | None
    if raw_output:
        middleware = RawPassthroughMiddleware()
    else:
        middleware = StreamPrinter()

    return_code, _stdout, _stderr = await adapter.run_container(
        image=image,
        volumes=volumes,
        environment=environment,
        command=command,
        middleware=middleware,
        user_context=user_context,
        ports=ports,
        extra_args=extra_args,
    )
    return return_code


@APP.command()
def build(
    image: str = typer.Option(
        DEFAULT_IMAGE, help="Target image tag (e.g., ccproxy:local)"
    ),
    context: Path = typer.Option(
        DEFAULT_CONTEXT,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing the Dockerfile (defaults to repo root).",
    ),
    no_cache: bool = typer.Option(False, help="Disable Docker build cache."),
    tty: bool = typer.Option(
        False,
        "--tty/--no-tty",
        help="Stream docker build output directly with terminal formatting.",
    ),
    dockerfile: Path | None = typer.Option(
        None,
        "--dockerfile",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Alternate Dockerfile to use for the build.",
    ),
) -> None:
    """Build the Docker image for ccproxy."""

    adapter = DockerAdapter(DockerConfig(docker_image=image))

    async def _runner() -> None:
        rc = await _build_image(
            adapter,
            image,
            context,
            no_cache,
            raw_output=tty,
            dockerfile=dockerfile,
        )
        if rc != 0:
            raise typer.Exit(rc)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        raise typer.Exit(130)


@APP.command()
def run(
    image: str = typer.Option(
        DEFAULT_IMAGE, help="Image tag to run (auto-builds if missing)."
    ),
    context: Path = typer.Option(
        DEFAULT_CONTEXT,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Docker build context for optional auto-build.",
    ),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p", help="Host port to expose."),
    container_port: int = typer.Option(
        DEFAULT_PORT, help="Container port to expose (defaults to 8000)."
    ),
    workspace: Path = typer.Option(
        PROJECT_ROOT,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Workspace to mount inside the container.",
    ),
    workspace_mount: str = typer.Option(
        DEFAULT_WORKSPACE_MOUNT,
        help="Container path for the workspace mount.",
    ),
    env: list[str] = typer.Option(
        [], "--env", "-e", help="Additional environment variables (KEY=VALUE)."
    ),
    apt_package: list[str] = typer.Option(
        [],
        "--apt-package",
        help="APT packages to install before starting (repeat or comma list).",
    ),
    volume: list[str] = typer.Option(
        [], "--volume", "-v", help="Additional volume mounts (HOST:CONTAINER)."
    ),
    init_script: Path | None = typer.Option(
        None,
        "--init-script",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Initialization script executed inside the container before CMD.",
    ),
    dockerfile: Path | None = typer.Option(
        None,
        "--dockerfile",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Alternate Dockerfile to use when auto-building the image.",
    ),
    enable_sudo: bool = typer.Option(
        False,
        "--enable-sudo/--no-enable-sudo",
        help="Grant container user passwordless sudo access.",
    ),
    cmd: str | None = typer.Option(
        None,
        "--cmd",
        help="Override container command (defaults to image CMD).",
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Disable auto-build when the image is missing."
    ),
    no_cache: bool = typer.Option(False, help="Disable cache if auto-build kicks in."),
    user_mapping: bool = typer.Option(
        False,
        "--user-mapping/--no-user-mapping",
        help="Map container user to the current host UID/GID.",
    ),
    tty: bool = typer.Option(
        False,
        "--tty/--no-tty",
        help="Attach an interactive TTY (-it) when starting the container.",
    ),
    command: list[str] = typer.Argument(
        [],
        metavar="CMD...",
        help="Command to execute inside the container (provide after --).",
    ),
) -> None:
    """Run the ccproxy container with helpful mounts."""

    adapter = DockerAdapter(
        DockerConfig(
            docker_image=image,
            user_mapping_enabled=user_mapping,
            user_uid=os.getuid() if user_mapping else None,
            user_gid=os.getgid() if user_mapping else None,
        )
    )

    workspace_path = workspace.expanduser().resolve()
    additional_env = _parse_env(env)
    apt_packages = _parse_apt_packages(apt_package)
    additional_volumes = _parse_volumes(volume)
    default_volumes = _default_mounts(workspace_path, workspace_mount)

    extra_volumes: list[tuple[str, str]] = []
    init_script_env_path: str | None = None
    if init_script is not None:
        resolved_script = init_script.expanduser().resolve()
        if not resolved_script.is_file():
            raise typer.BadParameter("Initialization script must be a regular file.")
        try:
            relative = resolved_script.relative_to(workspace_path)
        except ValueError:
            init_script_env_path = CONTAINER_INIT_SCRIPT
            extra_volumes.append((str(resolved_script), init_script_env_path))
        else:
            init_script_env_path = str(Path(workspace_mount) / relative)

    all_volumes = default_volumes + additional_volumes + extra_volumes

    environment: dict[str, str] = {
        "SERVER__HOST": "0.0.0.0",
        "SERVER__PORT": str(container_port),
        "XDG_CONFIG_HOME": f"{CONTAINER_HOME}/.config",
        "XDG_CACHE_HOME": f"{CONTAINER_HOME}/.cache",
        "CLAUDE_HOME": CONTAINER_CLAUDE,
        "CLAUDE_WORKSPACE": workspace_mount,
        "HOME": CONTAINER_HOME,
    }

    environment["PUID"] = str(os.getuid())
    environment["PGID"] = str(os.getgid())

    if apt_packages:
        environment["APT_PACKAGE"] = ",".join(apt_packages)

    if init_script_env_path:
        environment["INIT_SCRIPT"] = init_script_env_path

    if enable_sudo:
        environment["ENABLE_SUDO"] = "1"

    if tty:
        term = os.environ.get("TERM")
        if term:
            environment.setdefault("TERM", term)
        colorterm = os.environ.get("COLORTERM")
        if colorterm:
            environment.setdefault("COLORTERM", colorterm)
        color_env = os.environ.get("FORCE_COLOR")
        if color_env:
            environment.setdefault("FORCE_COLOR", color_env)

    environment.update(additional_env)
    command_list = command if command else (shlex.split(cmd) if cmd else None)
    ports = [f"{port}:{container_port}"]

    # final_command = command_list if command_list else ["ccproxy"]
    final_command = []
    docker_args: list[str] = []

    if tty:
        if os.isatty(0) and os.isatty(1):
            docker_args.extend(["-it"])
        else:
            typer.echo(
                "Warning: --tty requested but current stdin/stdout are not TTYs.",
                err=True,
            )

    command_list = final_command

    async def _runner() -> None:
        await _ensure_image(
            adapter,
            image=image,
            context=context,
            no_cache=no_cache,
            auto_build=not no_build,
            raw_output=tty,
            dockerfile=dockerfile,
        )
        rc = await _run_container(
            adapter,
            image=image,
            command=command_list,
            volumes=all_volumes,
            environment=environment,
            ports=ports,
            user_mapping=user_mapping,
            extra_args=docker_args,
            raw_output=tty,
        )
        if rc != 0:
            raise typer.Exit(rc)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        typer.echo("Interrupted, stopping container...", err=True)
        raise typer.Exit(130)


@APP.command()
def extend(
    output_image: str = typer.Option(..., help="Tag for the derived image."),
    base_image: str = typer.Option(
        DEFAULT_IMAGE,
        help="Base image to extend (builds automatically if missing).",
    ),
    context: Path = typer.Option(
        DEFAULT_CONTEXT,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Docker build context used when auto-building the base image.",
    ),
    apt_package: list[str] = typer.Option(
        [],
        "--apt-package",
        help="APT packages to install in the derived image (repeatable).",
    ),
    setup_script: Path | None = typer.Option(
        None,
        "--setup-script",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional host script executed during build.",
    ),
    setup_command: str | None = typer.Option(
        None,
        "--setup-command",
        help="Shell commands run during build (`bash -lc`).",
    ),
    no_build: bool = typer.Option(
        False,
        "--no-build",
        help="Disable auto-building the base image if it is missing.",
    ),
    no_cache: bool = typer.Option(
        False,
        help="Disable Docker build cache for the derived image.",
    ),
    tty: bool = typer.Option(
        False,
        "--tty/--no-tty",
        help="Stream docker build output directly with terminal formatting.",
    ),
) -> None:
    """Create a derived image that layers packages or setup steps."""

    if not (apt_package or setup_script or setup_command):
        raise typer.BadParameter(
            "Provide at least one --apt-package, --setup-script, or --setup-command"
        )

    adapter = DockerAdapter(DockerConfig(docker_image=base_image))

    async def _runner() -> None:
        await _ensure_image(
            adapter,
            image=base_image,
            context=context,
            no_cache=no_cache,
            auto_build=not no_build,
            raw_output=tty,
        )

        output_repo, output_tag = _split_image_reference(output_image)

        with tempfile.TemporaryDirectory(prefix="ccproxy-docker-extend-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            dockerfile_lines: list[str] = [f"FROM {base_image}"]

            if apt_package:
                packages = " ".join(shlex.quote(pkg) for pkg in apt_package)
                dockerfile_lines.append("ENV DEBIAN_FRONTEND=noninteractive")
                dockerfile_lines.append(
                    f"RUN apt-get update && apt-get install -y {packages} && rm -rf /var/lib/apt/lists/*"
                )

            if setup_script:
                build_target = "docker-runner-bootstrap.sh"
                destination = tmp_path / build_target
                shutil.copyfile(setup_script, destination)
                dockerfile_lines.append("RUN mkdir -p /docker-runner")
                dockerfile_lines.append(
                    f"COPY {build_target} /docker-runner/{build_target}"
                )
                dockerfile_lines.append(
                    f"RUN chmod +x /docker-runner/{build_target} \\\n    && /docker-runner/{build_target} \\\n    && rm /docker-runner/{build_target}"
                )

            if setup_command:
                dockerfile_lines.append("RUN bash -lc " + shlex.quote(setup_command))

            dockerfile_content = "\n".join(dockerfile_lines) + "\n"
            (tmp_path / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

            middleware: OutputMiddleware[Any] | None
            if tty:
                middleware = RawPassthroughMiddleware()
            else:
                middleware = StreamPrinter()

            rc = await adapter.build_image(
                dockerfile_dir=tmp_path,
                image_name=output_repo,
                image_tag=output_tag,
                no_cache=no_cache,
                middleware=middleware,
            )
            if rc != 0:
                raise typer.Exit(rc)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        raise typer.Exit(130)


if __name__ == "__main__":
    APP()
