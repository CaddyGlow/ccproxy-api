"""Hook implementation for command replay generation."""

from ccproxy.core.logging import get_plugin_logger
from ccproxy.core.plugins.hooks import Hook
from ccproxy.core.plugins.hooks.base import HookContext
from ccproxy.core.plugins.hooks.events import HookEvent
from ccproxy.core.request_context import RequestContext
from ccproxy.utils.command_line import (
    format_command_output,
    generate_curl_command,
    generate_xh_command,
)

from .config import CommandReplayConfig
from .formatter import CommandFileFormatter


logger = get_plugin_logger(__name__)


class CommandReplayHook(Hook):
    """Hook for generating curl and xh command replays of provider requests.

    Listens for PROVIDER_REQUEST_PREPARED events and generates command line
    equivalents that can be used to replay the exact same HTTP requests.
    """

    name = "command_replay"
    events = [
        HookEvent.PROVIDER_REQUEST_PREPARED,
        # Also listen to HTTP_REQUEST for broader coverage
        HookEvent.HTTP_REQUEST,
    ]
    priority = 200  # Run after core tracing but before heavy processing

    def __init__(
        self,
        config: CommandReplayConfig | None = None,
        file_formatter: CommandFileFormatter | None = None,
    ) -> None:
        """Initialize the command replay hook.

        Args:
            config: Command replay configuration
            file_formatter: File formatter for writing commands to files
        """
        self.config = config or CommandReplayConfig()
        self.file_formatter = file_formatter

        logger.debug(
            "command_replay_hook_initialized",
            enabled=self.config.enabled,
            generate_curl=self.config.generate_curl,
            generate_xh=self.config.generate_xh,
            include_patterns=self.config.include_url_patterns,
            only_provider_requests=self.config.only_provider_requests,
            include_client_requests=self.config.include_client_requests,
            write_to_files=self.config.write_to_files,
            log_dir=self.config.log_dir,
        )

    async def __call__(self, context: HookContext) -> None:
        """Handle hook events for command replay generation.

        Args:
            context: Hook context with event data
        """
        if not self.config.enabled:
            return

        # Debug logging
        logger.debug(
            "command_replay_hook_called",
            hook_event=context.event.value if context.event else "unknown",
            data_keys=list(context.data.keys()) if context.data else [],
        )

        try:
            if context.event == HookEvent.PROVIDER_REQUEST_PREPARED:
                await self._handle_provider_request(context)
            elif context.event == HookEvent.HTTP_REQUEST:
                await self._handle_http_request(context)
        except Exception as e:
            logger.error(
                "command_replay_hook_error",
                hook_event=context.event.value if context.event else "unknown",
                error=str(e),
                exc_info=e,
            )

    async def _handle_provider_request(self, context: HookContext) -> None:
        """Handle PROVIDER_REQUEST_PREPARED event."""
        await self._generate_commands(context, is_provider_request=True)

    async def _handle_http_request(self, context: HookContext) -> None:
        """Handle HTTP_REQUEST event - for both provider and client requests."""
        url = context.data.get("url", "")
        is_provider = self._is_provider_request(url)

        # Apply filtering based on configuration
        if self.config.only_provider_requests and not is_provider:
            return

        if not self.config.include_client_requests and not is_provider:
            return

        await self._generate_commands(context, is_provider_request=is_provider)

    async def _generate_commands(
        self, context: HookContext, is_provider_request: bool = False
    ) -> None:
        """Generate curl and xh commands from request context.

        Args:
            context: Hook context with request data
            is_provider_request: Whether this came from PROVIDER_REQUEST_PREPARED
        """
        # Extract request data
        method = context.data.get("method", "GET")
        url = context.data.get("url", "")
        headers = context.data.get("headers", {})
        body = context.data.get("body")
        is_json = context.data.get("is_json", False)

        # Get request ID for correlation
        request_id = (
            context.data.get("request_id")
            or context.metadata.get("request_id")
            or "unknown"
        )

        # Get provider name if available
        provider = context.provider or self._extract_provider_from_url(url)

        # Check if we should generate commands for this URL
        if not self.config.should_generate_for_url(url, is_provider_request):
            logger.debug(
                "command_replay_skipped_url_filter",
                request_id=request_id,
                url=url,
                provider=provider,
                is_provider_request=is_provider_request,
            )
            return

        # Validate we have minimum required data
        if not url or not method:
            logger.warning(
                "command_replay_insufficient_data",
                request_id=request_id,
                has_url=bool(url),
                has_method=bool(method),
            )
            return

        commands = []

        # Generate curl command
        if self.config.generate_curl:
            try:
                curl_cmd = generate_curl_command(
                    method=method,
                    url=url,
                    headers=headers,
                    body=body,
                    is_json=is_json,
                    pretty=self.config.pretty_format,
                )
                commands.append(("curl", curl_cmd))
            except Exception as e:
                logger.error(
                    "command_replay_curl_generation_error",
                    request_id=request_id,
                    error=str(e),
                )

        # Generate xh command
        if self.config.generate_xh:
            try:
                xh_cmd = generate_xh_command(
                    method=method,
                    url=url,
                    headers=headers,
                    body=body,
                    is_json=is_json,
                    pretty=self.config.pretty_format,
                )
                commands.append(("xh", xh_cmd))
            except Exception as e:
                logger.error(
                    "command_replay_xh_generation_error",
                    request_id=request_id,
                    error=str(e),
                )

        # Process generated commands
        if commands:
            curl_cmd = next((cmd for tool, cmd in commands if tool == "curl"), "")
            xh_cmd = next((cmd for tool, cmd in commands if tool == "xh"), "")

            # Write to files if enabled
            written_files = []
            if self.config.write_to_files and self.file_formatter:
                try:
                    # Get timestamp prefix from current request context if available
                    timestamp_prefix = None
                    try:
                        current_context = RequestContext.get_current()
                        if current_context:
                            timestamp_prefix = (
                                current_context.get_log_timestamp_prefix()
                            )
                    except Exception:
                        pass

                    written_files = await self.file_formatter.write_commands(
                        request_id=request_id,
                        curl_command=curl_cmd,
                        xh_command=xh_cmd,
                        provider=provider,
                        timestamp_prefix=timestamp_prefix,
                        method=method,
                        url=url,
                        headers=headers,
                        body=body,
                        is_json=is_json,
                    )

                    if written_files:
                        logger.debug(
                            "command_replay_files_written",
                            request_id=request_id,
                            files=written_files,
                            provider=provider,
                        )
                except Exception as e:
                    logger.error(
                        "command_replay_file_write_failed",
                        request_id=request_id,
                        error=str(e),
                        exc_info=e,
                    )

            # Log to console if enabled
            if self.config.log_to_console:
                output = format_command_output(
                    request_id=request_id,
                    curl_command=curl_cmd,
                    xh_command=xh_cmd,
                    provider=provider,
                )

                # Add file info to console output if files were written
                if written_files:
                    output += f"\n📁 Files written: {', '.join(written_files)}\n"

                logger.debug("command_replay_generated", output=output)

    def _is_provider_request(self, url: str) -> bool:
        """Determine if this is a request to a provider API.

        Args:
            url: The request URL

        Returns:
            True if this is a provider request
        """
        provider_domains = [
            "api.anthropic.com",
            "claude.ai",
            "api.openai.com",
            "chatgpt.com",
        ]

        url_lower = url.lower()
        return any(domain in url_lower for domain in provider_domains)

    def _extract_provider_from_url(self, url: str) -> str | None:
        """Extract provider name from URL.

        Args:
            url: The request URL

        Returns:
            Provider name or None if not recognized
        """
        url_lower = url.lower()

        if "anthropic.com" in url_lower or "claude.ai" in url_lower:
            return "anthropic"
        elif "openai.com" in url_lower or "chatgpt.com" in url_lower:
            return "openai"

        return None
