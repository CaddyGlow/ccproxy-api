"""Claude SDK exceptions."""


class ClaudeSDKError(Exception):
    """Base Claude SDK error."""

    pass


class StreamTimeoutError(ClaudeSDKError):
    """Stream timeout error when no SDK message is received within timeout."""

    def __init__(self, message: str, session_id: str, timeout_seconds: float):
        super().__init__(message)
        self.session_id = session_id
        self.timeout_seconds = timeout_seconds
