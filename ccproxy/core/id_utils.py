"""Utilities for generating short, debug-friendly IDs."""

import uuid


# Length of generated IDs - easily adjustable
ID_LENGTH = 8


def generate_short_id() -> str:
    """Generate a short, debug-friendly ID.

    Creates an 8-character hex string from a UUID4, providing good
    collision resistance while being much easier to type and remember
    during debugging.

    Returns:
        Short hex string (e.g., 'f47ac10b')
    """
    return uuid.uuid4().hex[:ID_LENGTH]
