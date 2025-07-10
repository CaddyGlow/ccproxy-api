"""CORS middleware for Claude Code Proxy API Server."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ccproxy.config.settings import Settings
from ccproxy.core.logging import get_logger


logger = get_logger(__name__)


def setup_cors_middleware(app: FastAPI, settings: Settings) -> None:
    """Setup CORS middleware for the FastAPI application.

    Args:
        app: FastAPI application instance
        settings: Application settings containing CORS configuration
    """
    logger.debug("Setting up CORS middleware")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_credentials,
        allow_methods=settings.cors_methods,
        allow_headers=settings.cors_headers,
        allow_origin_regex=settings.cors_origin_regex,
        expose_headers=settings.cors_expose_headers,
        max_age=settings.cors_max_age,
    )

    logger.info(f"CORS middleware configured: origins={settings.cors_origins}")


def get_cors_config(settings: Settings) -> dict:
    """Get CORS configuration dictionary.

    Args:
        settings: Application settings containing CORS configuration

    Returns:
        Dictionary containing CORS configuration
    """
    return {
        "allow_origins": settings.cors_origins,
        "allow_credentials": settings.cors_credentials,
        "allow_methods": settings.cors_methods,
        "allow_headers": settings.cors_headers,
        "allow_origin_regex": settings.cors_origin_regex,
        "expose_headers": settings.cors_expose_headers,
        "max_age": settings.cors_max_age,
    }
