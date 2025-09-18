"""Registry and decorators for LLM formatter functions."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


FormatterKey = tuple[str, str, str]

# Built-in helper modules that register formatters on import
# Import builtin helper modules at module import time so their
# decorators execute immediately (no lazy import).
# Defer importing helper modules until after decorator and registry are defined
# to avoid circular import issues during module initialization. We still avoid
# lazy imports by triggering these imports at the end of this module's load.
_DEFERRED_HELPER_IMPORTS = (
    "ccproxy.llms.formatters.anthropic_to_openai.helpers",
    "ccproxy.llms.formatters.openai_to_anthropic.helpers",
    "ccproxy.llms.formatters.openai_to_openai.helpers",
    "ccproxy.llms.formatters.response_api.helpers",
)


def register_formatter(
    source_format: str,
    target_format: str,
    operation: str,
    formatter: Callable[..., Any],
    *,
    module_name: str | None = None,
) -> None:
    """No-op placeholder retained for compatibility.

    Decorated functions are consumed directly by import side-effects.
    """
    return None


def formatter(
    source_format: str, target_format: str, operation: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records a formatter as soon as it is defined."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        register_formatter(source_format, target_format, operation, func)
        return func

    return decorator


def iter_registered_formatters() -> Iterable[tuple[str, str, str]]:
    """Deprecated: returns empty iterable (no central registry)."""
    return ()


def load_builtin_formatter_modules() -> None:
    """No-op retained for backward compatibility.

    Builtin formatter helper modules are now imported at module load time
    to avoid lazy imports. This function remains to keep existing call sites
    harmless.
    """
    return None


class FormatterRegistration:  # Deprecated compatibility alias
    pass


__all__ = [
    "FormatterKey",
    "FormatterRegistration",
    "formatter",
    "iter_registered_formatters",
    "load_builtin_formatter_modules",
    "register_formatter",
]

# Trigger import of builtin helper modules at the end of module initialization
# so decorator definitions above are available, avoiding circular imports while
# still not relying on per-call lazy imports.
try:  # pragma: no cover - defensive import
    for _mod in _DEFERRED_HELPER_IMPORTS:
        __import__(_mod)
except Exception as _exc:  # pragma: no cover - defensive logging
    logger.warning(
        "formatter_module_import_failed",
        module=str(_mod),
        error=str(_exc),
        category="formatter",
    )
