"""Compatibility gates between the installed Claude CLI and Agent SDK."""

import pytest
from claude_agent_sdk import RateLimitEvent
from claude_agent_sdk._internal.message_parser import parse_message


@pytest.mark.unit
def test_current_cli_rate_limit_event_is_non_terminal_wire_data() -> None:
    """Usage telemetry must not abort an otherwise valid SDK response stream."""

    message = parse_message({
        "type": "rate_limit_event",
        "uuid": "event-1",
        "session_id": "session-1",
        "rate_limit_info": {
            "status": "allowed",
            "rateLimitType": "five_hour",
            "utilization": 0.25,
        },
    })

    assert isinstance(message, RateLimitEvent)
    assert message.rate_limit_info.status == "allowed"
