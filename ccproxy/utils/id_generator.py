"""Utility functions for generating consistent IDs across the application."""

import uuid


def generate_client_id() -> str:
    """Generate a consistent client ID for SDK connections.

    Returns:
        str: First part of a UUID4 (8 characters)
    """
    return str(uuid.uuid4()).split("-")[0]
