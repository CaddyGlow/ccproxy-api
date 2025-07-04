from pathlib import Path
from typing import Any

from claude_code_sdk import ClaudeCodeOptions


def get_package_dir():
    try:
        import importlib.util

        # Get the path to the claude_code_proxy package and resolve it
        spec = importlib.util.find_spec("claude_code_proxy")
        if spec and spec.origin:
            package_dir = Path(spec.origin).parent.parent.resolve()
        else:
            package_dir = Path(__file__).parent.parent.parent.resolve()
    except Exception:
        package_dir = Path(__file__).parent.parent.parent.resolve()

    return package_dir
