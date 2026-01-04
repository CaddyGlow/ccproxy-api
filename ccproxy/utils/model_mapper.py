"""Utilities for applying provider-specific model mapping rules."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from ccproxy.models.provider import ModelMappingRule


_FLAG_MAP = {
    "IGNORECASE": re.IGNORECASE,
}


@dataclass(slots=True)
class MappingMatch:
    """Result of applying a mapping rule to a model identifier."""

    original: str
    mapped: str
    rule: ModelMappingRule | None


class ModelMapper:
    """Apply ordered mapping rules to model identifiers."""

    def __init__(self, rules: Sequence[ModelMappingRule] | None = None) -> None:
        self._rules = list(rules or [])
        self._compiled = [self._compile_rule(rule) for rule in self._rules]

    @staticmethod
    def _compile_rule(rule: ModelMappingRule) -> re.Pattern[str] | None:
        if rule.kind != "regex":
            return None
        flags = 0
        for flag_name in rule.flags:
            try:
                flags |= _FLAG_MAP[flag_name]
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise ValueError(f"Unsupported regex flag: {flag_name}") from exc
        return re.compile(rule.match, flags)

    @property
    def has_rules(self) -> bool:
        return bool(self._rules)

    def map(self, model_name: str) -> MappingMatch:
        """Return mapped model and matching rule, or passthrough if none match."""
        for rule, compiled in zip(self._rules, self._compiled, strict=False):
            if self._matches(rule, compiled, model_name):
                return MappingMatch(original=model_name, mapped=rule.target, rule=rule)
        return MappingMatch(original=model_name, mapped=model_name, rule=None)

    def _matches(
        self,
        rule: ModelMappingRule,
        compiled: re.Pattern[str] | None,
        model_name: str,
    ) -> bool:
        if rule.kind == "exact":
            return model_name == rule.match
        if rule.kind == "prefix":
            return model_name.startswith(rule.match)
        if rule.kind == "suffix":
            return model_name.endswith(rule.match)
        if rule.kind == "regex":
            assert compiled is not None
            return compiled.search(model_name) is not None
        return False

    def iter_rules(self) -> Iterable[ModelMappingRule]:
        """Expose rules for diagnostics and testing."""
        return iter(self._rules)


__all__ = [
    "ModelMapper",
    "MappingMatch",
    "add_model_alias",
    "restore_model_aliases",
]


_ALIAS_METADATA_KEY = "_model_alias_map"


def add_model_alias(metadata: dict[str, object], original: str, mapped: str) -> None:
    """Record a model alias mapping on the request metadata."""

    if original == mapped:
        return

    alias_map = metadata.setdefault(_ALIAS_METADATA_KEY, {})
    if isinstance(alias_map, dict):
        alias_map[mapped] = original


def restore_model_aliases(payload: object, metadata: Mapping[str, object]) -> object:
    """Restore aliased model identifiers in response payloads."""

    alias_map = metadata.get(_ALIAS_METADATA_KEY)
    if not isinstance(alias_map, Mapping) or not alias_map:
        return payload

    _restore_models(payload, alias_map)
    return payload


def _restore_models(obj: object, alias_map: Mapping[object, object]) -> None:
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == "model" and isinstance(value, str) and value in alias_map:
                obj[key] = alias_map[value]
            else:
                _restore_models(value, alias_map)
    elif isinstance(obj, list):
        for item in obj:
            _restore_models(item, alias_map)
