"""Utilities for validating format adapter chains during application startup."""

from __future__ import annotations

from fastapi import FastAPI

from ccproxy.core.logging import TraceBoundLogger
from ccproxy.services.adapters.chain_validation import (
    validate_chains,
    validate_stream_pairs,
)
from ccproxy.services.adapters.format_registry import FormatRegistry


def collect_declared_format_chains(app: FastAPI) -> list[list[str]]:
    """Collect declared format chains from FastAPI routes."""

    chains: list[list[str]] = []
    for route in app.router.routes:
        endpoint = getattr(route, "endpoint", None)
        chain = getattr(endpoint, "__format_chain__", None)
        if chain:
            chains.append(chain)
    return chains


def validate_route_format_chains(
    *,
    app: FastAPI,
    registry: FormatRegistry,
    logger: TraceBoundLogger,
) -> None:
    """Validate format chains declared on routes against the format registry."""

    try:
        declared_chains = collect_declared_format_chains(app)
        if not declared_chains:
            return

        missing = validate_chains(registry=registry, chains=declared_chains)
        missing_stream = validate_stream_pairs(
            registry=registry, chains=declared_chains
        )
        if missing or missing_stream:
            logger.error(
                "format_chain_validation_failed",
                missing_adapters=missing,
                missing_stream_adapters=missing_stream,
            )
    except Exception as exc:  # pragma: no cover - defensive logging path
        logger.warning("format_registry_setup_skipped", error=str(exc))


__all__ = ["validate_route_format_chains", "collect_declared_format_chains"]
