"""Utilities for generating command line tools (curl, xh) from HTTP request data."""

import json
import shlex
from typing import Any


def generate_curl_command(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: Any = None,
    is_json: bool = False,
    pretty: bool = True,
) -> str:
    """Generate a curl command from HTTP request parameters.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        headers: HTTP headers dictionary
        body: Request body (can be dict, str, bytes)
        is_json: Whether the body should be treated as JSON
        pretty: Whether to format the command for readability

    Returns:
        Complete curl command string
    """
    parts = ["curl"]

    # Add verbose flag for debugging
    parts.append("-v")

    # Add method if not GET
    if method.upper() != "GET":
        parts.extend(["-X", method.upper()])

    # Add headers
    if headers:
        for key, value in headers.items():
            parts.extend(["-H", f"{key}: {value}"])

    # Add body
    if isinstance(body, bytes):
        body_str = body.decode()
        parts.extend(["-d", body_str])

    # Add URL (always last)
    parts.append(url)

    if pretty:
        # Format for readability with line continuations
        cmd_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                cmd_parts.append(part)
            elif part in ["-X", "-H", "-d"]:
                cmd_parts.append(f" \\\n  {part}")
            elif i == len(parts) - 1:  # URL
                cmd_parts.append(f" \\\n  {shlex.quote(part)}")
            else:
                cmd_parts.append(f" {shlex.quote(part)}")
        return "".join(cmd_parts)
    else:
        # Single line, properly quoted
        return " ".join(shlex.quote(part) for part in parts)


def generate_xh_command(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: Any = None,
    is_json: bool = False,
    pretty: bool = True,
) -> str:
    """Generate an xh (HTTPie-like) command from HTTP request parameters.

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        headers: HTTP headers dictionary
        body: Request body (can be dict, str, bytes)
        is_json: Whether the body should be treated as JSON
        pretty: Whether to format the command for readability

    Returns:
        Complete xh command string
    """
    parts = ["xh"]

    # Add verbose flag for debugging
    parts.append("--verbose")

    # Add method and URL
    parts.append(f"{method.upper()}")
    parts.append(url)

    # Add headers
    if headers:
        for key, value in headers.items():
            # Quote the entire header to handle special characters and spaces
            parts.append(f"{key}:{value}")

    # Add body
    if isinstance(body, bytes):
        body_str = body.decode()
        parts.extend(["-d", body_str])

    if pretty:
        # Format for readability with line continuations
        cmd_parts = []
        for i, part in enumerate(parts):
            if i == 0:
                cmd_parts.append(part)
            elif part == "--verbose" or i == 1:
                cmd_parts.append(f" {part}")
            elif i == 2:  # URL
                cmd_parts.append(f" \\\n  {shlex.quote(part)}")
            elif part in ("--raw", "-d"):  # flags
                cmd_parts.append(f" \\\n  {part}")
            elif ":" in part and not part.startswith("http"):  # header
                cmd_parts.append(f" \\\n  {shlex.quote(part)}")
            else:
                cmd_parts.append(f" {shlex.quote(part)}")
        return "".join(cmd_parts)
    else:
        # Single line, properly quoted
        return " ".join(shlex.quote(part) for part in parts)


def generate_curl_shell_script(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: Any = None,
    is_json: bool = False,
) -> str:
    """Generate a shell script with curl command using proper JSON handling.

    This creates a more robust shell script that handles JSON safely by:
    1. Storing JSON in a variable using heredoc or printf
    2. Using the variable in the curl command

    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        headers: HTTP headers dictionary
        body: Request body (can be dict, str, bytes)
        is_json: Whether the body should be treated as JSON

    Returns:
        Complete shell script content
    """
    script_lines = ["#!/bin/bash", "set -e", ""]

    # Process JSON body safely
    json_data = None
    if body is not None and (is_json or isinstance(body, dict)):
        if isinstance(body, dict):
            json_data = json.dumps(
                body, indent=2, separators=(",", ": "), ensure_ascii=False
            )
        else:
            # Clean up string body
            body_str = str(body)
            if (body_str.startswith("b'") and body_str.endswith("'")) or (
                body_str.startswith('b"') and body_str.endswith('"')
            ):
                body_str = body_str[2:-1]

            body_str = body_str.replace('\\"', '"').replace("\\'", "'")

            try:
                parsed = json.loads(body_str)
                json_data = json.dumps(
                    parsed, indent=2, separators=(",", ": "), ensure_ascii=False
                )
            except (json.JSONDecodeError, ValueError):
                json_data = body_str

    # Build curl command parts
    curl_parts = ["curl", "-v"]

    if method.upper() != "GET":
        curl_parts.extend(["-X", shlex.quote(method.upper())])

    # Add headers
    if headers:
        for key, value in headers.items():
            curl_parts.extend(["-H", shlex.quote(f"{key}: {value}")])

    # Handle JSON body with heredoc
    if json_data:
        script_lines.append("# JSON payload")
        script_lines.append("JSON_DATA=$(cat <<'EOF'")
        script_lines.append(json_data)
        script_lines.append("EOF")
        script_lines.append(")")
        script_lines.append("")

        curl_parts.extend(["-d", '"$JSON_DATA"'])

        # Add content-type if not present
        if not headers or not any(k.lower() == "content-type" for k in headers):
            curl_parts.extend(["-H", shlex.quote("Content-Type: application/json")])
    elif body is not None:
        # Non-JSON body
        curl_parts.extend(["-d", shlex.quote(str(body))])

    # Add URL
    curl_parts.append(shlex.quote(url))

    # Build final command
    script_lines.append("# Execute curl command")
    script_lines.append(" ".join(curl_parts))
    script_lines.append("")

    return "\n".join(script_lines)


def format_command_output(
    request_id: str,
    curl_command: str,
    xh_command: str,
    provider: str | None = None,
) -> str:
    """Format the command output for logging.

    Args:
        request_id: Request ID for correlation
        curl_command: Generated curl command
        xh_command: Generated xh command
        provider: Provider name (optional)

    Returns:
        Formatted output string
    """
    provider_info = f" ({provider})" if provider else ""

    return f"""
🔄 Request Replay Commands{provider_info} [ID: {request_id}]

📋 curl:
{curl_command}

📋 xh:
{xh_command}

─────────────────────────────────────────────────────────────────────
"""
