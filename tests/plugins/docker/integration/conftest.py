"""Docker integration test fixtures.

Provides isolated, fast fixtures that mock Docker process execution so these
tests run without a real Docker daemon. Follows TESTING.md guidelines:
- Mock only external process boundaries
- Keep types explicit and fixtures minimal
- Mark tests with appropriate categories via test modules
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from ccproxy.plugins.docker.adapter import DockerAdapter
from ccproxy.plugins.docker.docker_path import DockerPath, DockerPathSet
from ccproxy.plugins.docker.models import DockerUserContext
from ccproxy.plugins.docker.stream_process import DefaultOutputMiddleware


@pytest.fixture
def docker_adapter_success(monkeypatch: pytest.MonkeyPatch) -> DockerAdapter:
    """DockerAdapter with successful mocked execution paths.

    - `is_available` returns True
    - `run_command` returns (0, ["ok"], []) for any command
    - `image_exists` returns True without invoking subprocess
    - `asyncio.create_subprocess_exec` returns a zero-returncode mock if
      exercised indirectly
    """
    adapter = DockerAdapter()

    # Force availability positive
    monkeypatch.setattr(adapter, "is_available", AsyncMock(return_value=True))

    # Patch run_command used by _run_with_sudo_fallback
    import ccproxy.plugins.docker.adapter as adapter_mod

    async def _ok_run_command(
        *_: object, **__: object
    ) -> tuple[int, list[str], list[str]]:
        return 0, ["ok"], []

    monkeypatch.setattr(adapter_mod, "run_command", _ok_run_command)

    # Ensure any direct subprocess execs in adapter code paths look successful
    async def _ok_proc_factory(*args: object, **kwargs: object):  # noqa: ANN001
        proc = Mock()
        proc.returncode = 0

        # .communicate used in is_available/image_exists paths
        async def _communicate() -> tuple[bytes, bytes]:
            return b"docker 25.0.0", b""

        proc.communicate = AsyncMock(side_effect=_communicate)
        # .wait used by stream runners
        proc.wait = AsyncMock(return_value=0)
        # .stdout/.stderr with readline for stream consumption
        stdout = AsyncMock()
        stdout.readline = AsyncMock(
            side_effect=[
                b"",
            ]
        )
        stderr = AsyncMock()
        stderr.readline = AsyncMock(
            side_effect=[
                b"",
            ]
        )
        proc.stdout = stdout
        proc.stderr = stderr
        return proc

    monkeypatch.setattr(adapter_mod.asyncio, "create_subprocess_exec", _ok_proc_factory)

    # Make image_exists trivially fast and deterministic
    monkeypatch.setattr(adapter, "image_exists", AsyncMock(return_value=True))

    return adapter


@pytest.fixture
def docker_adapter_failure(monkeypatch: pytest.MonkeyPatch) -> DockerAdapter:
    """DockerAdapter with failing mocked execution paths.

    - `is_available` returns True (so we reach execution)
    - `run_command` returns (1, [], ["error"]) for any command
    """
    adapter = DockerAdapter()
    monkeypatch.setattr(adapter, "is_available", AsyncMock(return_value=True))

    import ccproxy.plugins.docker.adapter as adapter_mod

    async def _err_run_command(
        *_: object, **__: object
    ) -> tuple[int, list[str], list[str]]:
        return 1, [], ["error"]

    monkeypatch.setattr(adapter_mod, "run_command", _err_run_command)
    return adapter


@pytest.fixture
def docker_adapter_unavailable(monkeypatch: pytest.MonkeyPatch) -> DockerAdapter:
    """DockerAdapter with Docker unavailable (is_available -> False)."""
    adapter = DockerAdapter()
    monkeypatch.setattr(adapter, "is_available", AsyncMock(return_value=False))
    return adapter


@pytest.fixture
def docker_user_context(tmp_path: Path) -> DockerUserContext:
    """Provide a deterministic DockerUserContext for tests."""
    home = DockerPath(host_path=tmp_path / "home", container_path="/data/home")
    workspace = DockerPath(
        host_path=tmp_path / "workspace", container_path="/data/workspace"
    )
    # Ensure directories exist for cleanliness
    home.host_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    workspace.host_path.mkdir(parents=True, exist_ok=True)  # type: ignore[union-attr]
    return DockerUserContext.create_manual(
        uid=1000,
        gid=1000,
        username="testuser",
        home_path=home,
        workspace_path=workspace,
        enable_user_mapping=True,
    )


@pytest.fixture
def docker_path_fixture(tmp_path: Path) -> DockerPath:
    """Single DockerPath mapping for tests."""
    host_dir = tmp_path / "host_dir"
    host_dir.mkdir(parents=True, exist_ok=True)
    return DockerPath(
        host_path=host_dir,
        container_path="/app/data",
        env_definition_variable_name="DATA_PATH",
    )


@pytest.fixture
def docker_path_set_fixture(tmp_path: Path) -> DockerPathSet:
    """DockerPathSet with two paths for integration tests."""
    base = tmp_path / "paths"
    base.mkdir(parents=True, exist_ok=True)
    paths = DockerPathSet(base_host_path=base)
    paths.add("data1", "/app/data1")
    paths.add("data2", "/app/data2")
    return paths


@pytest.fixture
def output_middleware() -> DefaultOutputMiddleware:
    """Default output middleware instance for stream processing tests."""
    return DefaultOutputMiddleware()
