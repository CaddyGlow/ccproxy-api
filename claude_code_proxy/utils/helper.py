from claude_code_sdk import ClaudeCodeOptions


def create_claude_code_options(
    self,
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    system: str | None = None,
    **kwargs: Any,
) -> ClaudeCodeOptions:
    """Create Claude Code SDK options from parameters."""
    effective_system = system or self.system_prompt

    return ClaudeCodeOptions(
        model=model or self.default_model,
        system_prompt=effective_system,
        max_turns=kwargs.get("max_turns", 1),
        permission_mode=kwargs.get("permission_mode", "default"),
        allowed_tools=kwargs.get("allowed_tools", []),
        disallowed_tools=kwargs.get("disallowed_tools", []),
    )
