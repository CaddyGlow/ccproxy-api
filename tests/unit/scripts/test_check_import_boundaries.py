from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from textwrap import dedent
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def checker_module() -> ModuleType:
    module_name = "_check_import_boundaries_test_module"
    script_path = (
        Path(__file__).resolve().parents[3] / "scripts" / "check_import_boundaries.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load check_import_boundaries script")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_reports_plugin_import_violation(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    core_dir = tmp_path / "ccproxy"
    target_file = core_dir / "core" / "foo.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text(
        dedent(
            """
            from ccproxy.plugins.example.plugin import factory

            def noop() -> None:
                pass
            """
        ).lstrip(),
        encoding="utf-8",
    )

    violations = checker_module.find_violations_in_file(
        target_file,
        core_dir,
        checker_module.DEFAULT_CONTEXT_LINES,
    )

    assert len(violations) == 1
    assert violations[0].display_line_number == 1
    assert "ccproxy.plugins.example" in violations[0].violating_line


def test_allowlist_skips_known_core_import(
    tmp_path: Path, checker_module: ModuleType
) -> None:
    core_dir = tmp_path / "ccproxy"
    allowed_file = core_dir / "core" / "plugins" / "discovery.py"
    allowed_file.parent.mkdir(parents=True)
    allowed_file.write_text(
        dedent(
            """
            import ccproxy.plugins


            def noop() -> None:
                pass
            """
        ).lstrip(),
        encoding="utf-8",
    )

    violations = checker_module.find_violations_in_file(
        allowed_file,
        core_dir,
        checker_module.DEFAULT_CONTEXT_LINES,
    )

    assert violations == []
