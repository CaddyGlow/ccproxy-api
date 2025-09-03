#!/usr/bin/env python3
"""Check import boundaries between core and plugins.

Rules:
- Core code under `ccproxy/` must not import from `plugins.*` modules.
- Allowed exceptions: code under `ccproxy/plugins/` itself (plugin framework),
  test files, and tooling/scripts.

Returns non-zero if violations are found.
"""

from __future__ import annotations

import pathlib
import re
import sys
from collections.abc import Iterable


CORE_DIR = pathlib.Path("ccproxy")
ALLOW_UNDER = {CORE_DIR / "plugins"}

IMPORT_PATTERN = re.compile(r"^(?:from|import)\s+plugins(\.|\s|$)")


def iter_py_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    for p in root.rglob("*.py"):
        # Skip hidden and cache dirs
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def is_allowed(path: pathlib.Path) -> bool:
    # Allow plugin framework code
    return any((path.resolve()).is_relative_to(allow) for allow in ALLOW_UNDER)


def main() -> int:
    if not CORE_DIR.exists():
        print("ccproxy/ not found; nothing to check")
        return 0

    violations: list[str] = []
    for file in iter_py_files(CORE_DIR):
        try:
            text = file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not IMPORT_PATTERN.search(text):
            continue

        # Ignore allowed subtrees
        if is_allowed(file):
            continue

        violations.append(str(file))

    if violations:
        print("Import boundary violations detected (core -> plugins.*):")
        for v in violations:
            print(f" - {v}")
        return 1

    print("Import boundaries OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
