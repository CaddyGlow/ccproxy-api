import json
import time
from datetime import datetime
from typing import Any


class AccessLogFormatter:
    """Format access logs for both client and provider levels.

    Supports Common Log Format, Combined Log Format, and Structured JSON.
    """

    def format_client(self, data: dict[str, Any], format_type: str) -> str:
        """Format client access log based on specified format.

        Args:
            data: Log data dictionary
            format_type: One of "common", "combined", or "structured"

        Returns:
            Formatted log line
        """
        if format_type == "common":
            return self._format_common(data)
        elif format_type == "combined":
            return self._format_combined(data)
        else:
            return self._format_structured_client(data)

    def format_provider(self, data: dict[str, Any]) -> str:
        """Format provider access log (always structured).

        Args:
            data: Log data dictionary

        Returns:
            JSON formatted log line
        """
        log_data = {
            "timestamp": data.get("timestamp"),
            "request_id": data.get("request_id"),
            "provider": data.get("provider"),
            "method": data.get("method"),
            "url": data.get("url"),
            "status_code": data.get("status_code"),
            "duration_ms": data.get("duration_ms"),
            "tokens_input": data.get("tokens_input"),
            "tokens_output": data.get("tokens_output"),
            "cache_read_tokens": data.get("cache_read_tokens"),
            "cache_write_tokens": data.get("cache_write_tokens"),
            "cost_usd": data.get("cost_usd"),
            "model": data.get("model"),
        }

        # Remove None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        return json.dumps(log_data)

    def _format_common(self, data: dict[str, Any]) -> str:
        """Format as Common Log Format.

        Format: host ident authuser date request status bytes
        Example: 127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326
        """
        timestamp = datetime.fromtimestamp(data.get("timestamp", time.time()))
        formatted_time = timestamp.strftime("%d/%b/%Y:%H:%M:%S %z")

        client_ip = data.get("client_ip", "-")
        method = data.get("method", "-")
        path = data.get("path", "")
        query = data.get("query", "")
        full_path = f"{path}?{query}" if query else path
        status = data.get("status_code", 0)
        bytes_sent = data.get("body_size", 0)

        # Use "-" for missing bytes field per Common Log Format spec
        bytes_str = str(bytes_sent) if bytes_sent > 0 else "-"

        return f'{client_ip} - - [{formatted_time}] "{method} {full_path} HTTP/1.1" {status} {bytes_str}'

    def _format_combined(self, data: dict[str, Any]) -> str:
        """Format as Combined Log Format.

        Format: Common + referer + user-agent
        Example: 127.0.0.1 - - [10/Oct/2000:13:55:36 -0700] "GET /apache_pb.gif HTTP/1.0" 200 2326 "http://www.example.com/start.html" "Mozilla/4.08 [en] (Win98; I ;Nav)"
        """
        common = self._format_common(data)

        # We don't typically have referer in API requests, use "-"
        referer = '"-"'

        # Get user agent or use "-"
        user_agent = data.get("user_agent", "-")
        user_agent_str = f'"{user_agent}"' if user_agent != "-" else '"-"'

        return f"{common} {referer} {user_agent_str}"

    def _format_structured_client(self, data: dict[str, Any]) -> str:
        """Format as structured JSON (matching existing access_logger.py).

        Includes all available fields for comprehensive logging.
        """
        log_data = {
            "timestamp": data.get("timestamp"),
            "request_id": data.get("request_id"),
            "method": data.get("method"),
            "path": data.get("path"),
            "query": data.get("query"),
            "status_code": data.get("status_code"),
            "duration_ms": data.get("duration_ms"),
            "client_ip": data.get("client_ip"),
            "user_agent": data.get("user_agent"),
            "body_size": data.get("body_size"),
            "error": data.get("error"),
            # These fields come from enriched context (if available)
            "endpoint": data.get("endpoint"),
            "model": data.get("model"),
            "service_type": data.get("service_type"),
            "tokens_input": data.get("tokens_input"),
            "tokens_output": data.get("tokens_output"),
            "cost_usd": data.get("cost_usd"),
        }

        # Remove None values for cleaner logs
        log_data = {k: v for k, v in log_data.items() if v is not None}
        return json.dumps(log_data)
