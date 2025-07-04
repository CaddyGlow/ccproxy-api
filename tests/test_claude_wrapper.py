"""Tests for Claude wrapper functionality."""

import json
import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from claude_code_proxy.utils.claude_wrapper import ClaudeWrapper, create_claude_wrapper
from claude_code_proxy.utils.subprocess_security import SubprocessSecurity


class TestClaudeWrapper:
    """Test suite for ClaudeWrapper class."""

    @pytest.fixture
    def mock_subprocess_security(self):
        """Create a mock SubprocessSecurity instance."""
        mock_security = MagicMock(spec=SubprocessSecurity)
        mock_security.secure_run = MagicMock()
        mock_security.secure_popen = MagicMock()
        return mock_security

    @pytest.fixture
    def claude_wrapper(self, mock_subprocess_security):
        """Create a ClaudeWrapper instance with mocked security."""
        return ClaudeWrapper(
            security=mock_subprocess_security, claude_path="/mock/claude"
        )

    @pytest.fixture
    def sample_status_json(self):
        """Sample JSON status response."""
        return {
            "status": "connected",
            "api_key": "set",
            "model": "claude-3-5-sonnet-20241022",
            "requests_today": 42,
            "last_request": "2024-01-15T10:30:00Z",
        }

    @pytest.fixture
    def sample_status_text(self):
        """Sample text status response."""
        return """
Configuration:
  API Key: Set (anthropic-key-*****xyz)
  Model: claude-3-5-sonnet-20241022
  Max Tokens: 4096
  Temperature: 1.0

Status:
  Connection: Connected
  Last Request: 2024-01-15T10:30:00Z
  Requests Today: 42

Memory Usage:
  Current: 128MB
  Peak: 256MB
"""

    @pytest.fixture
    def sample_interactive_output(self):
        """Sample interactive output with press enter prompt."""
        return [
            "Configuration loaded...\n",
            "API Key: Set\n",
            "Model: claude-3-5-sonnet-20241022\n",
            "Press Enter to continue...\n",
            "Additional status information:\n",
            "Connection: Established\n",
            "Requests today: 42\n",
        ]

    def test_init_default_settings(self):
        """Test ClaudeWrapper initialization with default settings."""
        wrapper = ClaudeWrapper()
        assert wrapper.claude_path is not None
        assert wrapper.security is not None
        assert wrapper._process is None

    def test_init_custom_settings(self, mock_subprocess_security):
        """Test ClaudeWrapper initialization with custom settings."""
        wrapper = ClaudeWrapper(
            security=mock_subprocess_security, claude_path="/custom/claude"
        )
        assert wrapper.claude_path == "/custom/claude"
        assert wrapper.security == mock_subprocess_security

    @patch("shutil.which")
    def test_find_claude_executable_in_path(self, mock_which):
        """Test finding claude executable in PATH."""
        mock_which.return_value = "/usr/bin/claude"
        wrapper = ClaudeWrapper()
        assert wrapper.claude_path == "/usr/bin/claude"

    def test_find_claude_executable_common_paths(self, mock_subprocess_security):
        """Test finding claude executable in common paths."""
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists") as mock_exists,
        ):
            # First path doesn't exist, second one does
            mock_exists.side_effect = [False, True]

            wrapper = ClaudeWrapper(security=mock_subprocess_security)
            assert wrapper.claude_path == "/usr/bin/claude"

    def test_find_claude_executable_fallback(self, mock_subprocess_security):
        """Test fallback when claude executable not found."""
        with (
            patch("shutil.which", return_value=None),
            patch("pathlib.Path.exists", return_value=False),
        ):
            wrapper = ClaudeWrapper(security=mock_subprocess_security)
            assert wrapper.claude_path == "claude"

    def test_execute_status_success(self, claude_wrapper, sample_status_json):
        """Test successful status execution."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_status_json)
        mock_result.stderr = ""

        claude_wrapper.security.secure_run.return_value = mock_result

        result = claude_wrapper.execute_status()

        assert result == sample_status_json
        claude_wrapper.security.secure_run.assert_called_once()
        args, kwargs = claude_wrapper.security.secure_run.call_args
        assert args[0] == ["/mock/claude", "/status"]
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        assert kwargs["timeout"] == 30

    def test_execute_status_command_failure(self, claude_wrapper):
        """Test status execution with command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Authentication failed"

        claude_wrapper.security.secure_run.return_value = mock_result

        with pytest.raises(
            RuntimeError, match="Claude command failed.*1.*Authentication failed"
        ):
            claude_wrapper.execute_status()

    def test_execute_status_subprocess_error(self, claude_wrapper):
        """Test status execution with subprocess error."""
        claude_wrapper.security.secure_run.side_effect = FileNotFoundError(
            "claude not found"
        )

        with pytest.raises(FileNotFoundError, match="claude not found"):
            claude_wrapper.execute_status()

    def test_parse_status_response_json(self, claude_wrapper, sample_status_json):
        """Test parsing JSON status response."""
        json_string = json.dumps(sample_status_json)
        result = claude_wrapper._parse_status_response(json_string)
        assert result == sample_status_json

    def test_parse_status_response_text(self, claude_wrapper, sample_status_text):
        """Test parsing text status response."""
        result = claude_wrapper._parse_status_response(sample_status_text)

        assert "configuration" in result
        assert "status" in result
        assert "memory_usage" in result

        config = result["configuration"]
        assert config["api_key"] == "Set (anthropic-key-*****xyz)"
        assert config["model"] == "claude-3-5-sonnet-20241022"
        assert config["max_tokens"] == "4096"
        assert config["temperature"] == "1.0"

        status = result["status"]
        assert status["connection"] == "Connected"
        assert status["last_request"] == "2024-01-15T10:30:00Z"
        assert status["requests_today"] == "42"

    def test_parse_status_response_invalid_json(self, claude_wrapper):
        """Test parsing invalid JSON response."""
        invalid_json = '{"invalid": json}'
        result = claude_wrapper._parse_status_response(invalid_json)

        assert result["parsed"] is False
        assert "error" in result
        assert result["raw_output"] == invalid_json

    def test_parse_status_response_empty(self, claude_wrapper):
        """Test parsing empty response."""
        result = claude_wrapper._parse_status_response("")

        assert result == {}

    @patch("subprocess.Popen")
    def test_execute_interactive_status_success(
        self, mock_popen, claude_wrapper, sample_interactive_output
    ):
        """Test successful interactive status execution."""
        mock_process = Mock()
        mock_process.poll.side_effect = [
            None,
            None,
            None,
            None,
            0,
        ]  # Process ends after 4 iterations
        mock_process.stdout.readline.side_effect = sample_interactive_output + [""]
        mock_process.wait.return_value = None
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_process.stdin = Mock()

        claude_wrapper.security.secure_popen.return_value = mock_process

        with patch("builtins.print") as mock_print:
            result = claude_wrapper.execute_interactive_status()

        # Verify process was created correctly
        claude_wrapper.security.secure_popen.assert_called_once()
        args, kwargs = claude_wrapper.security.secure_popen.call_args
        assert args[0] == ["/mock/claude", "/status"]
        assert kwargs["stdin"] == subprocess.PIPE
        assert kwargs["stdout"] == subprocess.PIPE
        assert kwargs["stderr"] == subprocess.PIPE
        assert kwargs["text"] is True

        # Verify Enter was sent when prompt detected
        mock_process.stdin.write.assert_called_with("\n")
        mock_process.stdin.flush.assert_called()

        # Verify output was printed
        assert mock_print.call_count > 0

        # Verify result structure
        assert isinstance(result, dict)
        # The result should be parsed from the output
        full_output = "".join(sample_interactive_output)
        assert any(key in result for key in ["raw_output", "output", "api_key"])

    @patch("subprocess.Popen")
    def test_execute_interactive_status_process_failure(
        self, mock_popen, claude_wrapper
    ):
        """Test interactive status execution with process failure."""
        mock_process = Mock()
        mock_process.poll.return_value = 1  # Process failed
        mock_process.stdout.readline.return_value = ""
        mock_process.wait.return_value = None
        mock_process.communicate.return_value = ("", "Error occurred")
        mock_process.returncode = 1

        claude_wrapper.security.secure_popen.return_value = mock_process

        with pytest.raises(
            RuntimeError, match="Claude command failed.*1.*Error occurred"
        ):
            claude_wrapper.execute_interactive_status()

    def test_execute_command_status_non_interactive(
        self, claude_wrapper, sample_status_json
    ):
        """Test execute_command with /status (non-interactive)."""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(sample_status_json)
        mock_result.stderr = ""

        claude_wrapper.security.secure_run.return_value = mock_result

        result = claude_wrapper.execute_command("/status", interactive=False)

        assert result == sample_status_json

    def test_execute_command_status_interactive(
        self, claude_wrapper, sample_interactive_output
    ):
        """Test execute_command with /status (interactive)."""
        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, None, None, 0]
        mock_process.stdout.readline.side_effect = sample_interactive_output + [""]
        mock_process.wait.return_value = None
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_process.stdin = Mock()

        claude_wrapper.security.secure_popen.return_value = mock_process

        with patch("builtins.print"):
            result = claude_wrapper.execute_command("/status", interactive=True)

        assert isinstance(result, dict)
        # The result should be parsed from the output
        full_output = "".join(sample_interactive_output)
        assert any(key in result for key in ["raw_output", "output", "api_key"])

    def test_execute_command_help_non_interactive(self, claude_wrapper):
        """Test execute_command with /help (non-interactive)."""
        help_output = "Claude CLI Help:\n  /status - Show status\n  /help - Show help"

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = help_output
        mock_result.stderr = ""

        claude_wrapper.security.secure_run.return_value = mock_result

        result = claude_wrapper.execute_command("/help", interactive=False)

        assert result["output"] == help_output
        assert result["command"] == "/help"
        assert result["interactive"] is False

    @patch("subprocess.Popen")
    def test_execute_command_help_interactive(self, mock_popen, claude_wrapper):
        """Test execute_command with /help (interactive)."""
        help_output = [
            "Claude CLI Help:\n",
            "  /status - Show status\n",
            "  /help - Show help\n",
        ]

        mock_process = Mock()
        mock_process.poll.side_effect = [None, None, None, 0]
        mock_process.stdout.readline.side_effect = help_output + [""]
        mock_process.wait.return_value = None
        mock_process.communicate.return_value = ("", "")
        mock_process.returncode = 0
        mock_process.stdin = Mock()

        claude_wrapper.security.secure_popen.return_value = mock_process

        with patch("builtins.print"):
            result = claude_wrapper.execute_command("/help", interactive=True)

        assert result["output"] == "".join(help_output)
        assert result["command"] == "/help"
        assert result["interactive"] is True

    def test_execute_command_failure(self, claude_wrapper):
        """Test execute_command with command failure."""
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Unknown command"

        claude_wrapper.security.secure_run.return_value = mock_result

        with pytest.raises(
            RuntimeError, match="Claude command failed.*1.*Unknown command"
        ):
            claude_wrapper.execute_command("/unknown")

    @patch("signal.signal")
    def test_setup_signal_handlers(self, mock_signal, claude_wrapper):
        """Test signal handler setup."""
        claude_wrapper._setup_signal_handlers()

        # Verify signal handlers were set up
        assert mock_signal.call_count == 2
        calls = mock_signal.call_args_list

        # Check SIGINT handler
        assert calls[0][0][0] == 2  # signal.SIGINT
        assert callable(calls[0][0][1])

        # Check SIGTERM handler
        assert calls[1][0][0] == 15  # signal.SIGTERM
        assert callable(calls[1][0][1])

    def test_signal_handler_with_process(self, claude_wrapper):
        """Test signal handler behavior with active process."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.terminate.return_value = None
        mock_process.wait.return_value = None

        claude_wrapper._process = mock_process

        with patch("sys.exit") as mock_exit:
            claude_wrapper._setup_signal_handlers()
            # Get the signal handler function
            import signal

            with patch("signal.signal") as mock_signal:
                claude_wrapper._setup_signal_handlers()
                signal_handler = mock_signal.call_args_list[0][0][1]

                # Call the signal handler
                signal_handler(2, None)

                # Verify process was terminated
                mock_process.terminate.assert_called_once()
                mock_process.wait.assert_called_once_with(timeout=5)
                mock_exit.assert_called_once_with(0)

    def test_signal_handler_process_kill(self, claude_wrapper):
        """Test signal handler with process that needs killing."""
        mock_process = Mock()
        mock_process.poll.return_value = None  # Process is running
        mock_process.terminate.return_value = None
        mock_process.wait.side_effect = subprocess.TimeoutExpired("cmd", 5)
        mock_process.kill.return_value = None

        claude_wrapper._process = mock_process

        with patch("sys.exit") as mock_exit:
            claude_wrapper._setup_signal_handlers()
            import signal

            with patch("signal.signal") as mock_signal:
                claude_wrapper._setup_signal_handlers()
                signal_handler = mock_signal.call_args_list[0][0][1]

                # Call the signal handler
                signal_handler(2, None)

                # Verify process was terminated then killed
                mock_process.terminate.assert_called_once()
                mock_process.wait.assert_called_once_with(timeout=5)
                mock_process.kill.assert_called_once()
                mock_exit.assert_called_once_with(0)


class TestCreateClaudeWrapper:
    """Test suite for create_claude_wrapper convenience function."""

    def test_create_claude_wrapper_default(self):
        """Test creating wrapper with default settings."""
        wrapper = create_claude_wrapper()
        assert isinstance(wrapper, ClaudeWrapper)
        assert wrapper.claude_path is not None

    def test_create_claude_wrapper_custom_claude_path(self):
        """Test creating wrapper with custom claude path."""
        wrapper = create_claude_wrapper(claude_path="/custom/claude")
        assert wrapper.claude_path == "/custom/claude"

    def test_create_claude_wrapper_custom_security(self):
        """Test creating wrapper with custom security settings."""
        # Test that parameters are passed through correctly
        wrapper = create_claude_wrapper(
            claude_path="/test/claude",
            working_directory="/tmp/test",
            user="testuser",
            group="testgroup",
        )

        # Verify the wrapper was created with correct claude path
        assert wrapper.claude_path == "/test/claude"
        # Security object should be created (even if it fails validation in tests)

    def test_create_claude_wrapper_no_security_params(self):
        """Test creating wrapper without security parameters."""
        wrapper = create_claude_wrapper(claude_path="/test/claude")
        assert wrapper.claude_path == "/test/claude"
        # Should use default security from get_default_claude_security


class TestClaudeWrapperIntegration:
    """Integration tests for ClaudeWrapper."""

    @pytest.fixture
    def temp_claude_script(self, tmp_path):
        """Create a temporary claude script for testing."""
        script_path = tmp_path / "claude"
        script_content = """#!/bin/bash
if [[ "$1" == "/status" ]]; then
    echo "Status: Connected"
    echo "API Key: Set"
    echo "Model: claude-3-5-sonnet-20241022"
    if [[ "$2" == "--interactive" ]]; then
        echo "Press Enter to continue..."
        read
        echo "Additional status info"
    fi
    exit 0
elif [[ "$1" == "/help" ]]; then
    echo "Claude CLI Help:"
    echo "  /status - Show status"
    echo "  /help - Show help"
    exit 0
else
    echo "Unknown command: $1" >&2
    exit 1
fi
"""
        script_path.write_text(script_content)
        script_path.chmod(0o755)
        return str(script_path)

    @pytest.mark.integration
    @patch("claude_code_proxy.utils.subprocess_security.get_default_claude_security")
    def test_real_status_execution(self, mock_security, temp_claude_script):
        """Test real status execution with temporary script."""
        mock_security.return_value = Mock()
        mock_security.return_value.secure_run = lambda cmd, **kwargs: subprocess.run(
            cmd, **kwargs
        )

        wrapper = ClaudeWrapper(claude_path=temp_claude_script)

        result = wrapper.execute_status()

        assert "status" in result
        assert result["status"] == "Connected"
        assert "api_key" in result
        assert result["api_key"] == "Set"
        assert "model" in result
        assert result["model"] == "claude-3-5-sonnet-20241022"

    @pytest.mark.integration
    @patch("claude_code_proxy.utils.subprocess_security.get_default_claude_security")
    def test_real_help_execution(self, mock_security, temp_claude_script):
        """Test real help execution with temporary script."""
        mock_security.return_value = Mock()
        mock_security.return_value.secure_run = lambda cmd, **kwargs: subprocess.run(
            cmd, **kwargs
        )

        wrapper = ClaudeWrapper(claude_path=temp_claude_script)

        result = wrapper.execute_command("/help")

        assert "Claude CLI Help:" in result["output"]
        assert "/status - Show status" in result["output"]
        assert "/help - Show help" in result["output"]

    @pytest.mark.integration
    @patch("claude_code_proxy.utils.subprocess_security.get_default_claude_security")
    def test_real_command_failure(self, mock_security, temp_claude_script):
        """Test real command failure with temporary script."""
        mock_security.return_value = Mock()
        mock_security.return_value.secure_run = lambda cmd, **kwargs: subprocess.run(
            cmd, **kwargs
        )

        wrapper = ClaudeWrapper(claude_path=temp_claude_script)

        with pytest.raises(
            RuntimeError, match="Claude command failed.*1.*Unknown command"
        ):
            wrapper.execute_command("/unknown")


# Mark slow tests
pytest.mark.slow = pytest.mark.integration
