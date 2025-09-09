#!/usr/bin/env python3
"""Check import boundaries between core and plugins.

Rules:
- Core code under `ccproxy/` must not import from `plugins.*` modules.
- Allowed exceptions: code under `ccproxy/plugins/` itself (plugin framework),
  test files, and tooling/scripts.

Returns non-zero if violations are found.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple


DEFAULT_CONTEXT_LINES = 4  # Default number of context lines to show around violations


@dataclass
class ImportViolation:
    """Represents a single import boundary violation."""

    file: pathlib.Path
    line_number: int  # 0-based
    context_lines: list[str]
    context_line_count: int  # Number of context lines used

    @property
    def display_line_number(self) -> int:
        """1-based line number for display."""
        return self.line_number + 1

    @property
    def violating_line(self) -> str:
        """The actual line that contains the violation."""
        context_start = max(0, self.line_number - self.context_line_count)
        relative_index = self.line_number - context_start
        return (
            self.context_lines[relative_index]
            if 0 <= relative_index < len(self.context_lines)
            else ""
        )


class ImportInfo(NamedTuple):
    """Parsed import information from a line."""

    type: str  # "import_from" or "import"
    from_part: str
    import_part: str
    full_part: str


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Check import boundaries between core and plugins"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output violations as JSON lines (machine-readable)",
    )
    parser.add_argument(
        "--context-lines",
        "-n",
        type=int,
        default=DEFAULT_CONTEXT_LINES,
        help=f"Number of context lines to show around violations (default: {DEFAULT_CONTEXT_LINES})",
    )
    return parser.parse_args()


def find_ccproxy_directory() -> pathlib.Path:
    """Find the ccproxy package directory dynamically."""
    spec = importlib.util.find_spec("ccproxy")
    if spec is None or not spec.submodule_search_locations:
        print("Could not find ccproxy module in the current environment.")
        sys.exit(1)
    return pathlib.Path(spec.submodule_search_locations[0])


# Pattern to match imports from ccproxy.plugins, allowing leading whitespace
IMPORT_PATTERN = re.compile(
    r"^\s*(?:from|import)\s+ccproxy\.plugins(\.|\s|$)",
    re.MULTILINE,
)


def iter_py_files(root: pathlib.Path) -> Iterable[pathlib.Path]:
    """Iterate over all Python files in the given directory, excluding hidden dirs."""
    for p in root.rglob("*.py"):
        # Skip hidden and cache dirs
        if any(part.startswith(".") for part in p.parts):
            continue
        yield p


def should_check_file(file: pathlib.Path, core_dir: pathlib.Path) -> bool:
    """Check if a file should be analyzed for import violations."""
    # Exclude files under ccproxy/plugins (plugin framework itself)
    return not file.is_relative_to(core_dir / "plugins")


def get_context_lines(
    lines: list[str], violation_line: int, context_lines: int
) -> list[str]:
    """Get context lines around a violation."""
    start = max(0, violation_line - context_lines)
    end = min(len(lines), violation_line + context_lines + 1)
    return lines[start:end]


def find_violations_in_file(
    file: pathlib.Path, context_lines: int
) -> list[ImportViolation]:
    """Find all import violations in a single file."""
    try:
        lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return []

    violations = []
    for line_idx, line in enumerate(lines):
        if IMPORT_PATTERN.search(line):
            context = get_context_lines(lines, line_idx, context_lines)
            violations.append(
                ImportViolation(
                    file=file,
                    line_number=line_idx,
                    context_lines=context,
                    context_line_count=context_lines,
                )
            )

    return violations


def parse_import_line(line: str) -> ImportInfo:
    """Parse import information from a line of code."""
    from_match = re.match(r"\s*from\s+([\w\.]+)\s+import\s+([\w\.,\s]+)", line)
    import_match = re.match(r"\s*import\s+([\w\.]+)", line)

    if from_match:
        from_part = from_match.group(1)
        import_part = from_match.group(2).replace(" ", "")
        full_part = ",".join([from_part + "." + imp for imp in import_part.split(",")])
        return ImportInfo("import_from", from_part, import_part, full_part)

    elif import_match:
        import_part = import_match.group(1)
        return ImportInfo("import", "", import_part, import_part)

    else:
        return ImportInfo("", "", "", "")


def output_json_violation(violation: ImportViolation) -> None:
    """Output a single violation in JSON format."""
    context_start = max(0, violation.line_number - violation.context_line_count)
    relative_idx = violation.line_number - context_start

    before = violation.context_lines[:relative_idx]
    line_text = violation.violating_line
    after = (
        violation.context_lines[relative_idx + 1 :]
        if relative_idx + 1 < len(violation.context_lines)
        else []
    )

    import_info = parse_import_line(line_text)

    output = {
        "file": str(violation.file),
        "line": violation.display_line_number,
        "type": import_info.type,
        "from": import_info.from_part,
        "import": import_info.import_part,
        "full": import_info.full_part,
        "before": before,
        "line_text": line_text,
        "after": after,
    }
    print(json.dumps(output))


def output_human_violation(violation: ImportViolation, use_color: bool) -> None:
    """Output a single violation in human-readable format."""
    print(f"{violation.file}:{violation.display_line_number}")

    context_start = max(0, violation.line_number - violation.context_line_count)
    for rel_idx, context_line in enumerate(violation.context_lines):
        line_no = context_start + rel_idx + 1
        marker = ">>" if line_no == violation.display_line_number else "  "

        if line_no == violation.display_line_number and use_color:
            # Print violating line in red
            print(f"{marker} {line_no:4}: \033[31m{context_line}\033[0m")
        else:
            print(f"{marker} {line_no:4}: {context_line}")
    print()


def find_all_violations(
    core_dir: pathlib.Path, context_lines: int
) -> list[ImportViolation]:
    """Find all import violations in the core directory."""
    violations = []

    for file in iter_py_files(core_dir):
        if should_check_file(file, core_dir):
            violations.extend(find_violations_in_file(file, context_lines))

    return violations


def main() -> int:
    """Main entry point for the import boundary checker."""
    args = parse_args()
    use_color = sys.stdout.isatty() and not args.json

    core_dir = find_ccproxy_directory()
    if not core_dir.exists():
        print("ccproxy/ not found; nothing to check")
        return 0

    violations = find_all_violations(core_dir, args.context_lines)

    if violations:
        if args.json:
            for violation in violations:
                output_json_violation(violation)
        else:
            print("Import boundary violations detected:\n")
            for violation in violations:
                output_human_violation(violation, use_color)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
