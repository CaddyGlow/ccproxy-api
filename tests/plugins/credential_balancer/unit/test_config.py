"""Tests covering credential balancer configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ccproxy.plugins.credential_balancer.config import CredentialManager


def test_credential_manager_expands_user_directory(
    monkeypatch: Any, tmp_path: Path
) -> None:
    fake_home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(fake_home))

    cred = CredentialManager(
        manager_class="test.Manager",
        storage_class="test.Storage",
        file=Path("~/tokens/sample.json"),
    )

    assert cred.file == fake_home / "tokens" / "sample.json"


def test_credential_manager_expands_environment_variables(
    monkeypatch: Any, tmp_path: Path
) -> None:
    credentials_dir = tmp_path / "configured"
    monkeypatch.setenv("CREDENTIALS_DIR", str(credentials_dir))

    cred = CredentialManager(
        manager_class="test.Manager",
        storage_class="test.Storage",
        file=Path("${CREDENTIALS_DIR}/token.json"),
    )

    assert cred.file == credentials_dir / "token.json"
