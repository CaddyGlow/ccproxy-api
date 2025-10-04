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
from pathlib import Path
from typing import Iterable

import typer

from ccproxy.plugins.docker.adapter import DockerAdapter
from ccproxy.plugins.docker.config import DockerConfig
from ccproxy.plugins.docker.models import DockerUserContext
from ccproxy.plugins.docker.stream_process import OutputMiddleware

APP = typer.Typer(help="Build and run the ccproxy Docker image with standard mounts")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IMAGE = "ccproxy:local"
DEFAULT_PORT = 8000
DEFAULT_CONTEXT = PROJECT_ROOT
DEFAULT_WORKSPACE_MOUNT = "/workspace"


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
            raise typer.BadParameter(f"Invalid volume specification '{spec}'. Use host:container")
        host, container = spec.split(":", 1)
        host_path = _ensure_directory(Path(host))
        if not container:
            raise typer.BadParameter(f"Container path missing in volume '{spec}'")
        volumes.append((str(host_path), container))
    return volumes


def _default_mounts(workspace: Path, workspace_mount: str) -> list[tuple[str, str]]:
    """Return the default host/container mounts used for Docker runs."""
    mounts: list[tuple[str, str]] = []

    def add(host: Path, container: str) -> None:
        mounts.append((str(_ensure_directory(host)), container))

    add(workspace, workspace_mount)
    add(Path.home() / ".codex", "/root/.codex")
    add(Path.home() / ".claude", "/root/.claude")
    add(Path.home() / ".config/gh", "/root/.config/gh")
    add(Path.home() / ".config/ccproxy", "/root/.config/ccproxy")
    add(Path.home() / ".cache/ccproxy", "/root/.cache/ccproxy")

    return mounts


async def _build_image(
    adapter: DockerAdapter,
    image: str,
    context: Path,
    no_cache: bool,
) -> int:
    repo, tag = _split_image_reference(image)
    rc, _stdout, _stderr = await adapter.build_image(
        context,
        image_name=repo,
        image_tag=tag,
        no_cache=no_cache,
        middleware=StreamPrinter(),
    )
    return rc


async def _ensure_image(
    adapter: DockerAdapter,
    image: str,
    context: Path,
    no_cache: bool,
    auto_build: bool,
) -> None:
    repo, tag = _split_image_reference(image)
    if await adapter.image_exists(repo, tag):
        return
    if not auto_build:
        raise typer.BadParameter(
            f"Image '{image}' not found locally. Run 'docker_runner.py build' or remove --no-build.")
    typer.echo(f"Image '{image}' not found. Building from {context}...")
    rc = await _build_image(adapter, image, context, no_cache)
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

    return_code, _stdout, _stderr = await adapter.run_container(
        image=image,
        volumes=volumes,
        environment=environment,
        command=command,
        middleware=StreamPrinter(),
        user_context=user_context,
        ports=ports,
    )
    return return_code


@APP.command()
def build(
    image: str = typer.Option(DEFAULT_IMAGE, help="Target image tag (e.g., ccproxy:local)"),
    context: Path = typer.Option(
        DEFAULT_CONTEXT,
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing the Dockerfile (defaults to repo root).",
    ),
    no_cache: bool = typer.Option(False, help="Disable Docker build cache."),
) -> None:
    """Build the Docker image for ccproxy."""

    adapter = DockerAdapter(DockerConfig(docker_image=image))

    async def _runner() -> None:
        rc = await _build_image(adapter, image, context, no_cache)
        if rc != 0:
            raise typer.Exit(rc)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        raise typer.Exit(130)


@APP.command()
def run(
    image: str = typer.Option(DEFAULT_IMAGE, help="Image tag to run (auto-builds if missing)."),
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
    volume: list[str] = typer.Option(
        [], "--volume", "-v", help="Additional volume mounts (HOST:CONTAINER)."
    ),
    cmd: str | None = typer.Option(
        None,
        "--cmd",
        help="Override container command (defaults to image CMD).",
    ),
    no_build: bool = typer.Option(
        False, "--no-build", help="Disable auto-build when the image is missing."
    ),
    no_cache: bool = typer.Option(
        False, help="Disable cache if auto-build kicks in."
    ),
    user_mapping: bool = typer.Option(
        True,
        "--user-mapping/--no-user-mapping",
        help="Map container user to the current host UID/GID.",
    ),
    apt_package: list[str] = typer.Option(
        [],
        "--apt-package",
        help="APT package to install before starting the proxy (repeatable).",
    ),
    setup_script: Path | None = typer.Option(
        None,
        "--setup-script",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
        help="Optional host script to run inside the container before start.",
    ),
    setup_command: str | None = typer.Option(
        None,
        "--setup-command",
        help="Inline shell commands to run before starting the proxy.",
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

    additional_env = _parse_env(env)
    additional_volumes = _parse_volumes(volume)
    default_volumes = _default_mounts(workspace, workspace_mount)
    all_volumes = default_volumes + additional_volumes

    if apt_package and user_mapping:
        typer.echo(
            "Disabling user-mapping so apt packages can be installed as root...",
            err=True,
        )
        user_mapping = False

    environment: dict[str, str] = {
        "SERVER__HOST": "0.0.0.0",
        "SERVER__PORT": str(container_port),
        "XDG_CONFIG_HOME": "/root/.config",
        "XDG_CACHE_HOME": "/root/.cache",
        "CLAUDE_HOME": "/root/.claude",
        "CLAUDE_WORKSPACE": workspace_mount,
    }

    if user_mapping:
        environment["PUID"] = str(os.getuid())
        environment["PGID"] = str(os.getgid())

    if apt_package:
        environment.setdefault("DEBIAN_FRONTEND", "noninteractive")

    environment.update(additional_env)
    command_list = shlex.split(cmd) if cmd else None
    ports = [f"{port}:{container_port}"]

    script_container_path: str | None = None
    if setup_script:
        if not setup_script.is_file():
            raise typer.BadParameter("--setup-script must point to a file")
        script_container_path = f"/tmp/docker-runner/{setup_script.name}"
        all_volumes.append((str(setup_script), f"{script_container_path}:ro"))

    final_command = command_list if command_list else ["ccproxy"]
    setup_lines: list[str] = []

    if apt_package:
        quoted = " ".join(shlex.quote(pkg) for pkg in apt_package)
        setup_lines.append("apt-get update")
        setup_lines.append(f"apt-get install -y {quoted}")

    if script_container_path:
        setup_lines.append(f"chmod +x {shlex.quote(script_container_path)}")
        setup_lines.append(shlex.quote(script_container_path))

    if setup_command:
        setup_lines.append(setup_command)

    if setup_lines:
        setup_lines.insert(0, "set -euo pipefail")
        exec_line = "exec " + " ".join(shlex.quote(arg) for arg in final_command)
        setup_lines.append(exec_line)
        command_list = ["bash", "-lc", "\n".join(setup_lines)]
    else:
        command_list = final_command

    async def _runner() -> None:
        await _ensure_image(
            adapter,
            image=image,
            context=context,
            no_cache=no_cache,
            auto_build=not no_build,
        )
        rc = await _run_container(
            adapter,
            image=image,
            command=command_list,
            volumes=all_volumes,
            environment=environment,
            ports=ports,
            user_mapping=user_mapping,
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
        )

        output_repo, output_tag = _split_image_reference(output_image)

        with tempfile.TemporaryDirectory(prefix="ccproxy-docker-extend-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            dockerfile_lines: list[str] = [f"FROM {base_image}"]

            if apt_package:
                packages = " ".join(apt_package)
                dockerfile_lines.append("ENV DEBIAN_FRONTEND=noninteractive")
                dockerfile_lines.append(
                    "RUN apt-get update \\n+ && apt-get install -y "
                    + packages
                    + " \\\n+ && rm -rf /var/lib/apt/lists/*"
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
                    f"RUN chmod +x /docker-runner/{build_target} \\\n+ && /docker-runner/{build_target} \\\n+ && rm /docker-runner/{build_target}"
                )

            if setup_command:
                dockerfile_lines.append(
                    "RUN bash -lc " + shlex.quote(setup_command)
                )

            dockerfile_content = "\n".join(dockerfile_lines) + "\n"
            (tmp_path / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")

            rc = await adapter.build_image(
                dockerfile_dir=tmp_path,
                image_name=output_repo,
                image_tag=output_tag,
                no_cache=no_cache,
                middleware=StreamPrinter(),
            )
            if rc != 0:
                raise typer.Exit(rc)

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        raise typer.Exit(130)


if __name__ == "__main__":
    APP()
