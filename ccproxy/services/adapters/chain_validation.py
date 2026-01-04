from __future__ import annotations

from collections.abc import Iterable

from .format_registry import FormatRegistry


def validate_chains(
    *, registry: FormatRegistry, chains: Iterable[list[str]]
) -> list[str]:
    """Validate that all adjacent pairs in chains exist in the registry.

    Returns a list of human‑readable error strings for missing pairs.
    """
    errors: list[str] = []
    pairs_needed: set[tuple[str, str]] = set()
    for chain in chains:
        if len(chain) >= 2:
            for i in range(len(chain) - 1):
                pairs_needed.add((chain[i], chain[i + 1]))
    for src, dst in sorted(pairs_needed):
        if registry.get_if_exists(src, dst) is None:
            errors.append(f"Missing format adapter: {src} -> {dst}")
    return errors


def validate_stream_pairs(
    *, registry: FormatRegistry, chains: Iterable[list[str]]
) -> list[str]:
    """Validate reverse-direction pairs for streaming (provider→client)."""
    missing: list[str] = []
    for chain in chains:
        if len(chain) < 2:
            continue
        reverse_pairs = list(
            reversed([(chain[i + 1], chain[i]) for i in range(len(chain) - 1)])
        )
        for src, dst in reverse_pairs:
            if registry.get_if_exists(src, dst) is None:
                missing.append(f"Missing streaming adapter: {src} -> {dst}")
    return missing


__all__ = ["validate_chains", "validate_stream_pairs"]
