"""CORS middleware for Claude Code Proxy API Server."""

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from structlog import get_logger

from ccproxy.config.settings import Settings


logger = get_logger(__name__)


def setup_cors_middleware(app: FastAPI, settings: Settings) -> None:
    """Setup CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings: Application settings containing CORS configuration
    """
    logger.debug("cors_middleware_setup_start")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors.origins,
        allow_credentials=settings.cors.credentials,
        allow_methods=settings.cors.methods,
        allow_headers=settings.cors.headers,
        allow_origin_regex=settings.cors.origin_regex,
        expose_headers=settings.cors.expose_headers,
        max_age=settings.cors.max_age,
    )

    logger.debug("cors_middleware_configured", origins=settings.cors.origins)


def get_cors_config(settings: Settings) -> dict[str, Any]:
    """Get CORS configuration dictionary.

    Args:
        settings: Application settings containing CORS configuration

    Returns:
        Dictionary containing CORS configuration
    """
    return {
        "allow_origins": settings.cors.origins,
        "allow_credentials": settings.cors.credentials,
        "allow_methods": settings.cors.methods,
        "allow_headers": settings.cors.headers,
        "allow_origin_regex": settings.cors.origin_regex,
        "expose_headers": settings.cors.expose_headers,
        "max_age": settings.cors.max_age,
    }
