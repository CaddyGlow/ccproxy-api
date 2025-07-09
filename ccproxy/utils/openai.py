"""OpenAI-related utility functions shared across the codebase."""


def is_openai_request(path: str) -> bool:
    """Check if request path suggests OpenAI format.

    Args:
        path: Request path to check

    Returns:
        True if this is an OpenAI format request
    """
    return path.startswith("/openai/") or path.endswith("/chat/completions")
