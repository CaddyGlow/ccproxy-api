from typing import Any


def extract_request_headers(request: Any) -> dict[str, str]:
    """Extract headers from request as lowercase dict."""
    headers = {}
    try:
        if hasattr(request, "headers") and hasattr(request.headers, "raw"):
            for name_bytes, value_bytes in request.headers.raw:
                name = name_bytes.decode("latin-1").lower()
                value = value_bytes.decode("latin-1")
                headers[name] = value
        elif hasattr(request, "headers"):
            for name, value in request.headers.items():
                headers[name.lower()] = value
    except UnicodeDecodeError as e:
        # Log encoding errors but don't fail the request
        from ccproxy.core.logging import get_plugin_logger

        logger = get_plugin_logger()
        logger.warning("header_decode_error", error=str(e))
    except Exception as e:
        # Log unexpected errors for debugging
        from ccproxy.core.logging import get_plugin_logger

        logger = get_plugin_logger()
        logger.debug("header_extraction_fallback", error=str(e))
    return headers


def extract_response_headers(response: Any) -> dict[str, str]:
    """Extract headers from response as lowercase dict."""
    headers = {}
    try:
        if hasattr(response, "headers"):
            for name, value in response.headers.items():
                headers[name.lower()] = value
    except UnicodeDecodeError as e:
        # Log encoding errors but don't fail the response
        from ccproxy.core.logging import get_plugin_logger

        logger = get_plugin_logger()
        logger.warning("response_header_decode_error", error=str(e))
    except Exception as e:
        # Log unexpected errors for debugging
        from ccproxy.core.logging import get_plugin_logger

        logger = get_plugin_logger()
        logger.debug("response_header_extraction_fallback", error=str(e))
    return headers


def to_canonical_headers(headers: dict[str, str]) -> dict[str, str]:
    """Convert lowercase headers to canonical case for HTTP."""
    canonical_map = {
        "content-type": "Content-Type",
        "content-length": "Content-Length",
        "authorization": "Authorization",
        "user-agent": "User-Agent",
        "accept": "Accept",
        "x-api-key": "X-API-Key",
        "x-request-id": "X-Request-ID",
        "x-github-api-version": "X-GitHub-Api-Version",
        "copilot-integration-id": "Copilot-Integration-Id",
        "editor-version": "Editor-Version",
        "editor-plugin-version": "Editor-Plugin-Version",
        "session-id": "Session-ID",
        "chatgpt-account-id": "ChatGPT-Account-ID",
        "openai-beta": "OpenAI-Beta",
        "originator": "Originator",
        "version": "Version",
    }

    result = {}
    for key, value in headers.items():
        canonical_key = canonical_map.get(key)
        if canonical_key:
            result[canonical_key] = value
        else:
            # Title case for unknown headers
            result["-".join(word.capitalize() for word in key.split("-"))] = value

    return result


def filter_request_headers(
    headers: dict[str, str],
    additional_excludes: set[str] | None = None,
    preserve_auth: bool = False,
) -> dict[str, str]:
    """Filter headers, ensuring lowercase keys in result."""
    excludes = EXCLUDED_REQUEST_HEADERS.copy()

    if preserve_auth:
        excludes.discard("authorization")
        excludes.discard("x-api-key")

    if additional_excludes:
        excludes.update(additional_excludes)

    filtered = {}
    for key, value in headers.items():
        if key.lower() not in excludes:
            filtered[key.lower()] = value

    return filtered


def filter_response_headers(
    headers: dict[str, str],
    additional_excludes: set[str] | None = None,
) -> dict[str, str]:
    """Filter response headers, ensuring lowercase keys in result."""
    excludes = {
        # Hop-by-hop headers
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        # Other headers to exclude
        "content-encoding",
        "content-length",
    }

    if additional_excludes:
        excludes.update(additional_excludes)

    filtered = {}
    for key, value in headers.items():
        if key.lower() not in excludes:
            filtered[key.lower()] = value

    return filtered


# Keep existing EXCLUDED_REQUEST_HEADERS constant
EXCLUDED_REQUEST_HEADERS = {
    # Connection-related headers
    "host",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "te",
    "trailer",
    # Proxy headers
    "proxy-authenticate",
    "proxy-authorization",
    "x-forwarded-for",
    "x-forwarded-proto",
    "x-forwarded-host",
    "forwarded",
    # Encoding headers
    "accept-encoding",
    "content-encoding",
    # CORS headers
    "origin",
    "access-control-request-method",
    "access-control-request-headers",
    "access-control-allow-origin",
    "access-control-allow-methods",
    "access-control-allow-headers",
    "access-control-allow-credentials",
    "access-control-max-age",
    "access-control-expose-headers",
    # Auth headers (will be replaced)
    # we cleanup by precaution
    "authorization",
    "x-api-key",
    # Content length (will be recalculated)
    "content-length",
}
