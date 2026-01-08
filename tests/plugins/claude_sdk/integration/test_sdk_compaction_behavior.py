"""Integration tests for SDK compaction behavior with tool blocks.

These tests verify whether the Claude Agent SDK's automatic context compaction
creates orphaned tool_use/tool_result blocks that cause API errors.

Tests require:
- Claude CLI installed and authenticated
- Real API calls (uses actual tokens)
- Sufficient context to trigger auto-compaction

Run with: RUN_SDK_INTEGRATION=1 pytest tests/plugins/claude_sdk/integration/test_sdk_compaction_behavior.py -v -s
"""

import json
import os
from pathlib import Path
from typing import Any

import pytest


# Mark all tests in this module as requiring SDK integration
pytestmark = [
    pytest.mark.integration,
    pytest.mark.real_api,  # Uses real_api marker for tests requiring actual SDK/API calls
    pytest.mark.skipif(
        not os.environ.get("RUN_SDK_INTEGRATION"),
        reason="SDK integration tests require RUN_SDK_INTEGRATION=1",
    ),
]


class TestSessionFileOrphanAnalysis:
    """Analyze existing session JSONL files for orphaned tool blocks.

    This test doesn't make API calls - it inspects existing session files
    to look for evidence of orphaned tool_use/tool_result pairs.
    """

    def _extract_tool_blocks(
        self,
        msg: dict[str, Any],
        tool_uses: dict[str, dict[str, Any]],
        tool_results: dict[str, dict[str, Any]],
    ) -> None:
        """Extract tool_use and tool_result blocks from a message."""
        content = msg.get("content", [])
        if isinstance(content, str):
            return

        if not isinstance(content, list):
            return

        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        tool_uses[tool_id] = block
                elif block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        tool_results[tool_use_id] = block

    def _analyze_session_file(self, session_file: Path) -> dict[str, Any]:
        """Analyze a single session file for tool block orphans."""
        tool_uses: dict[str, dict[str, Any]] = {}
        tool_results: dict[str, dict[str, Any]] = {}
        message_count = 0
        compaction_events = []

        with session_file.open() as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                    message_count += 1

                    # Check for compaction boundary
                    if (
                        msg.get("type") == "system"
                        and msg.get("subtype") == "compact_boundary"
                    ):
                        compaction_events.append(
                            {
                                "line": line_num,
                                "metadata": msg.get("compact_metadata", {}),
                            }
                        )

                    # Extract tool blocks
                    self._extract_tool_blocks(msg, tool_uses, tool_results)

                except json.JSONDecodeError:
                    continue

        # Find orphans
        orphaned_results = set(tool_results.keys()) - set(tool_uses.keys())
        orphaned_uses = set(tool_uses.keys()) - set(tool_results.keys())

        return {
            "file": str(session_file),
            "message_count": message_count,
            "tool_use_count": len(tool_uses),
            "tool_result_count": len(tool_results),
            "compaction_events": compaction_events,
            "orphaned_results": list(orphaned_results),
            "orphaned_uses": list(orphaned_uses),
            "has_orphans": bool(orphaned_results or orphaned_uses),
        }

    @pytest.mark.unit
    def test_inspect_session_history_for_orphans(self) -> None:
        """Inspect session JSONL files for orphaned tool blocks.

        This test directly reads the session files to check for orphans
        without making API calls.
        """
        claude_projects = Path.home() / ".claude" / "projects"

        if not claude_projects.exists():
            pytest.skip("No Claude projects directory found at ~/.claude/projects")

        # Find all session files recursively
        session_files = list(claude_projects.rglob("*.jsonl"))

        if not session_files:
            pytest.skip("No session files found in ~/.claude/projects")

        # Sort by modification time, analyze most recent
        session_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Analyze up to 10 most recent sessions
        results = []
        sessions_with_orphans = []
        sessions_with_compaction = []

        for session_file in session_files[:10]:
            analysis = self._analyze_session_file(session_file)
            results.append(analysis)

            if analysis["has_orphans"]:
                sessions_with_orphans.append(analysis)
            if analysis["compaction_events"]:
                sessions_with_compaction.append(analysis)

        # Report findings
        print(f"\n{'=' * 60}")
        print("SESSION FILE ANALYSIS")
        print(f"{'=' * 60}")
        print(f"Total sessions analyzed: {len(results)}")
        print(f"Sessions with compaction events: {len(sessions_with_compaction)}")
        print(f"Sessions with orphaned tool blocks: {len(sessions_with_orphans)}")
        print()

        for r in results:
            status = "HAS ORPHANS" if r["has_orphans"] else "OK"
            compaction = (
                f"({len(r['compaction_events'])} compactions)"
                if r["compaction_events"]
                else ""
            )
            print(
                f"  [{status}] {Path(r['file']).name}: "
                f"{r['tool_use_count']} uses, {r['tool_result_count']} results {compaction}"
            )

            if r["has_orphans"]:
                if r["orphaned_results"]:
                    print(
                        f"    - Orphaned tool_results (no matching tool_use): {r['orphaned_results'][:3]}..."
                    )
                if r["orphaned_uses"]:
                    print(
                        f"    - Orphaned tool_uses (no matching tool_result): {r['orphaned_uses'][:3]}..."
                    )

        print()

        # Correlation analysis
        if sessions_with_compaction:
            compaction_with_orphans = [
                s for s in sessions_with_compaction if s["has_orphans"]
            ]
            print(
                f"Compaction correlation: {len(compaction_with_orphans)}/{len(sessions_with_compaction)} "
                f"sessions with compaction also have orphans"
            )

        # The test passes regardless - this is investigative
        # But we flag if we found concerning patterns
        if sessions_with_orphans:
            print(
                f"\nWARNING: Found {len(sessions_with_orphans)} session(s) with orphaned tool blocks!"
            )
            for s in sessions_with_orphans:
                if s["compaction_events"]:
                    print(
                        f"  - {Path(s['file']).name}: Compaction may have created orphans"
                    )


class TestSDKCompactionTrigger:
    """Tests that require actual SDK interactions to trigger compaction."""

    @pytest.mark.asyncio
    async def test_compaction_creates_orphans_hypothesis(self) -> None:
        """Test hypothesis: SDK compaction creates orphaned tool blocks.

        This test:
        1. Creates a session with tool_use/tool_result exchanges
        2. Fills context until auto-compaction triggers
        3. Verifies whether orphaned tool blocks appear

        Note: This test is expensive (uses many tokens) and slow.
        """
        # Import SDK components
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        import uuid

        # Claude CLI requires proper UUID format for --resume
        session_id = str(uuid.uuid4())
        num_turns = 20  # Start with fewer turns, increase if needed
        compaction_detected = False
        error_messages: list[str] = []

        print(f"\nStarting compaction test with session: {session_id}")
        print(f"Will attempt {num_turns} turns to trigger compaction")

        for turn in range(num_turns):
            # Create a prompt that generates substantial content
            # Include mention of tools to encourage tool-like patterns in response
            prompt = f"""Turn {turn + 1}/{num_turns}:

Please provide a detailed technical analysis (at least 500 words) about:
- System architecture patterns for turn {turn}
- Include code examples with detailed comments
- Discuss trade-offs and implementation considerations

This is for testing context window behavior."""

            options = ClaudeAgentOptions(
                resume=session_id if turn > 0 else None,
                max_turns=1,
            )

            try:
                messages_this_turn = []
                async for message in query(prompt=prompt, options=options):
                    messages_this_turn.append(message)

                    # Check for compaction boundary
                    if hasattr(message, "type") and message.type == "system":
                        if (
                            hasattr(message, "subtype")
                            and message.subtype == "compact_boundary"
                        ):
                            compaction_detected = True
                            print(f"\n*** COMPACTION DETECTED at turn {turn + 1} ***")
                            if hasattr(message, "compact_metadata"):
                                print(f"    Metadata: {message.compact_metadata}")

                print(
                    f"  Turn {turn + 1} completed: {len(messages_this_turn)} messages"
                )

            except Exception as e:
                error_str = str(e)
                error_messages.append(error_str)

                if "tool_use_id" in error_str.lower():
                    print(f"\n*** ORPHAN ERROR DETECTED at turn {turn + 1} ***")
                    print(f"    Error: {error_str}")

                    # This confirms the hypothesis
                    pytest.fail(
                        f"SDK compaction created orphaned tool blocks at turn {turn + 1}: {error_str}"
                    )
                else:
                    print(f"  Turn {turn + 1} error (non-orphan): {error_str[:100]}")

        # Summary
        print(f"\n{'=' * 60}")
        print("TEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"Turns completed: {num_turns}")
        print(f"Compaction detected: {compaction_detected}")
        print(f"Errors encountered: {len(error_messages)}")

        if not compaction_detected:
            print("\nNote: Compaction did not trigger. To trigger compaction:")
            print("  - Increase num_turns (currently {num_turns})")
            print("  - Use prompts that generate more output")
            print("  - Or manually check ~/.claude/projects for session files")

    @pytest.mark.asyncio
    async def test_manual_orphan_injection(self) -> None:
        """Test how SDK handles a session file with injected orphans.

        This test:
        1. Creates a session file with orphaned tool blocks
        2. Attempts to resume the session
        3. Observes whether SDK/API errors occur
        """
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError:
            pytest.skip("claude_agent_sdk not installed")

        # This test would require manipulating session files directly
        # which is risky and may not be reproducible
        pytest.skip("Manual orphan injection test not yet implemented")


class TestCompareSDKvsAPI:
    """Compare tool handling between SDK and direct API paths."""

    @pytest.mark.asyncio
    async def test_identical_request_different_path(self) -> None:
        """Send identical requests through SDK and API, compare behavior.

        This tests whether the SDK and API handle tool blocks identically
        when given the same input.
        """
        pytest.skip(
            "Comparative test not yet implemented - requires both adapters running"
        )
