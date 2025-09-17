"""Registry and decorators for LLM formatter functions."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import structlog


logger = structlog.get_logger(__name__)


FormatterKey = tuple[str, str, str]


@dataclass(frozen=True)
class FormatterRegistration:
    """Declarative formatter registration metadata."""

    source_format: str
    target_format: str
    operation: str
    formatter: Callable[..., Any]
    module_name: str

    @property
    def key(self) -> FormatterKey:
        return self.source_format, self.target_format, self.operation


_STATIC_REGISTRATIONS: list[FormatterRegistration] = []

# Built-in helper modules that register formatters on import
_BUILTIN_HELPER_MODULES: tuple[str, ...] = (
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
    """Record a formatter for later registry population."""

    if not source_format or not target_format or not operation:
        raise ValueError(
            "Formatter registration requires non-empty source, target, and operation"
        )

    registration = FormatterRegistration(
        source_format=source_format,
        target_format=target_format,
        operation=operation,
        formatter=formatter,
        module_name=module_name or formatter.__module__,
    )
    _STATIC_REGISTRATIONS.append(registration)


def formatter(
    source_format: str, target_format: str, operation: str
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that records a formatter as soon as it is defined."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        register_formatter(source_format, target_format, operation, func)
        return func

    return decorator


def iter_registered_formatters() -> Iterable[FormatterRegistration]:
    """Return a snapshot of all statically registered formatters."""

    return tuple(_STATIC_REGISTRATIONS)


def load_builtin_formatter_modules() -> None:
    """Import helper modules so that their decorators run."""

    for module_path in _BUILTIN_HELPER_MODULES:
        try:
            import_module(module_path)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(
                "formatter_module_import_failed",
                module=module_path,
                error=str(exc),
                category="formatter",
            )


class FormatterRegistry:
    """Registry for managing formatter callables keyed by format/operation."""

    def __init__(self) -> None:
        self._formatters: dict[FormatterKey, Callable[..., Any]] = {}
        self._registered_modules: set[str] = set()

    def register(
        self,
        source_format: str,
        target_format: str,
        operation: str,
        formatter: Callable[..., Any],
        module_name: str | None = None,
    ) -> None:
        key = (source_format, target_format, operation)
        if key in self._formatters:
            existing = self._formatters[key]
            raise ValueError(
                "Formatter already registered for {} -> {} ({}): existing={}, attempted={}".format(
                    source_format,
                    target_format,
                    operation,
                    getattr(existing, "__name__", repr(existing)),
                    getattr(formatter, "__name__", repr(formatter)),
                )
            )

        self._formatters[key] = formatter
        if module_name:
            self._registered_modules.add(module_name)

        logger.debug(
            "formatter_registered",
            source_format=source_format,
            target_format=target_format,
            operation=operation,
            formatter=getattr(formatter, "__name__", str(formatter)),
            module=module_name,
            category="formatter",
        )

    def register_many(self, registrations: Iterable[FormatterRegistration]) -> None:
        """Load multiple formatter registrations."""

        for registration in registrations:
            self.register(
                registration.source_format,
                registration.target_format,
                registration.operation,
                registration.formatter,
                registration.module_name,
            )

    def get(
        self, source_format: str, target_format: str, operation: str
    ) -> Callable[..., Any]:
        key = (source_format, target_format, operation)
        formatter = self._formatters.get(key)
        if formatter is None:
            available = ", ".join(
                f"{src}->{tgt}:{op}" for src, tgt, op in sorted(self._formatters)
            )
            raise ValueError(
                f"No formatter registered for {source_format}->{target_format}:{operation}. Available: {available}"
            )
        return formatter

    def get_all(
        self, source_format: str, target_format: str
    ) -> dict[str, Callable[..., Any]]:
        result: dict[str, Callable[..., Any]] = {}
        for (src, tgt, operation), formatter in self._formatters.items():
            if src == source_format and tgt == target_format:
                result[operation] = formatter

        if not result:
            available = {(src, tgt) for (src, tgt, _op) in self._formatters}
            raise ValueError(
                f"No formatters registered for {source_format}->{target_format}. Available pairs: {sorted(available)}"
            )

        return result

    def get_bidirectional(
        self, format_a: str, format_b: str
    ) -> dict[str, dict[str, Callable[..., Any]]]:
        result: dict[str, dict[str, Callable[..., Any]]] = {}
        with contextlib.suppress(ValueError):
            result[f"{format_a}_to_{format_b}"] = self.get_all(format_a, format_b)

        with contextlib.suppress(ValueError):
            result[f"{format_b}_to_{format_a}"] = self.get_all(format_b, format_a)

        if not result:
            raise ValueError(
                f"No formatters registered for either direction between {format_a} and {format_b}"
            )

        return result

    def list(self) -> list[str]:
        return [f"{src}->{tgt}:{op}" for src, tgt, op in sorted(self._formatters)]

    def get_registered_modules(self) -> set[str]:
        return set(self._registered_modules)


__all__ = [
    "FormatterKey",
    "FormatterRegistration",
    "FormatterRegistry",
    "formatter",
    "iter_registered_formatters",
    "load_builtin_formatter_modules",
    "register_formatter",
]
