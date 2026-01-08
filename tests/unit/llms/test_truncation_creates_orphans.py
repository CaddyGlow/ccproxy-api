"""Tests reproducing the bug where truncation creates orphaned tool blocks.

This module demonstrates the issue where:
1. Sanitization runs FIRST (removes existing orphaned tool blocks)
2. Truncation runs SECOND (can CREATE new orphaned blocks by removing messages)
3. The orphaned blocks hit the Anthropic API and cause errors like:
   - "unexpected tool_use_id found in tool_result blocks"
   - "tool_use ids were found without tool_result blocks"

The fix should ensure sanitization runs AFTER truncation, or both before AND after.
"""

from typing import Any

import pytest

from ccproxy.llms.formatters.openai_to_anthropic.requests import _sanitize_tool_results
from ccproxy.llms.utils import truncate_to_fit


Message = dict[str, Any]


class TestTruncationCreatesOrphanedBlocks:
    """Tests demonstrating the bug where truncation creates orphaned tool blocks."""

    def _create_tool_use_assistant_message(
        self,
        tool_id: str,
        tool_name: str = "read_file",
        input_data: dict[str, Any] | None = None,
    ) -> Message:
        """Create an assistant message with a tool_use block."""
        return {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": tool_name,
                    "input": input_data or {"path": "/some/file.txt"},
                }
            ],
        }

    def _create_tool_result_user_message(
        self, tool_use_id: str, result_content: str = "File contents here"
    ) -> Message:
        """Create a user message with a tool_result block."""
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                }
            ],
        }

    def _create_conversation_with_tool_calls(self, num_pairs: int) -> list[Message]:
        """Create a conversation with multiple tool_use/tool_result pairs.

        Each pair consists of:
        1. Assistant message with tool_use block
        2. User message with tool_result block
        """
        messages: list[Message] = []
        # Start with initial user message
        messages.append(
            {"role": "user", "content": "Please help me analyze some files."}
        )

        for i in range(num_pairs):
            tool_id = f"tool_{i:03d}"
            # Assistant calls a tool
            messages.append(
                self._create_tool_use_assistant_message(
                    tool_id=tool_id,
                    tool_name="read_file",
                    input_data={"path": f"/file_{i}.txt"},
                )
            )
            # User provides tool result with substantial content to use up tokens
            messages.append(
                self._create_tool_result_user_message(
                    tool_use_id=tool_id,
                    result_content=f"This is the content of file {i}. "
                    * 50,  # Make it substantial
                )
            )

        # End with a final user message
        messages.append({"role": "user", "content": "Now summarize all the files."})
        return messages

    def _count_orphaned_tool_results(self, messages: list[Message]) -> int:
        """Count tool_result blocks that don't have a matching tool_use in preceding message."""
        orphan_count = 0

        for i, msg in enumerate(messages):
            if msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue

            # Get valid tool_use IDs from preceding assistant message
            valid_ids = set()
            if i > 0 and messages[i - 1].get("role") == "assistant":
                prev_content = messages[i - 1].get("content", [])
                if isinstance(prev_content, list):
                    for block in prev_content:
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            valid_ids.add(block.get("id"))

            # Check for orphaned tool_results
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    if block.get("tool_use_id") not in valid_ids:
                        orphan_count += 1

        return orphan_count

    def _count_orphaned_tool_uses(self, messages: list[Message]) -> int:
        """Count tool_use blocks that don't have a matching tool_result in any subsequent message."""
        # Collect all tool_result IDs
        result_ids = set()
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        result_ids.add(block.get("tool_use_id"))

        # Count tool_use blocks without results
        orphan_count = 0
        for msg in messages:
            if msg.get("role") == "assistant" and isinstance(msg.get("content"), list):
                for block in msg["content"]:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        if block.get("id") not in result_ids:
                            orphan_count += 1

        return orphan_count

    def test_current_order_creates_orphans(self) -> None:
        """REPRODUCTION TEST: Current order (sanitize → truncate) creates orphaned blocks.

        This test demonstrates the bug:
        1. Create a conversation with many tool_use/tool_result pairs
        2. Apply sanitization first (no orphans to remove initially)
        3. Apply truncation with low token limit (removes old messages)
        4. VERIFY: Orphaned tool_result blocks now exist (BUG!)

        The Anthropic API would reject this with:
        "unexpected tool_use_id found in tool_result blocks"
        """
        # Create conversation with 10 tool call pairs
        messages = self._create_conversation_with_tool_calls(num_pairs=10)

        # Verify no orphans initially
        initial_orphan_results = self._count_orphaned_tool_results(messages)
        initial_orphan_uses = self._count_orphaned_tool_uses(messages)
        assert initial_orphan_results == 0, "Should have no orphaned results initially"
        assert initial_orphan_uses == 0, "Should have no orphaned uses initially"

        # STEP 1: Sanitization (current order - runs first)
        sanitized_messages = _sanitize_tool_results(messages)

        # No orphans should be removed since there are none
        assert self._count_orphaned_tool_results(sanitized_messages) == 0

        # STEP 2: Truncation (current order - runs second)
        request = {
            "model": "claude-3-opus-20240229",
            "messages": sanitized_messages,
        }

        # Use a very low token limit to force significant truncation
        truncated_request, was_truncated = truncate_to_fit(
            request,
            max_input_tokens=2000,  # Very low to force truncation
            preserve_recent=4,  # Keep last 4 messages
            safety_margin=0.9,
        )

        assert was_truncated, "Should have been truncated"
        truncated_messages = truncated_request["messages"]

        # The bug: truncation removed assistant messages with tool_use blocks,
        # but the corresponding tool_result blocks in user messages remain
        orphan_count = self._count_orphaned_tool_results(truncated_messages)

        # This assertion documents the bug - we EXPECT orphans due to the wrong order
        # When the fix is applied, this test should be updated
        print(
            f"\n[BUG REPRODUCTION] Found {orphan_count} orphaned tool_result blocks after truncation"
        )
        print(f"Messages before truncation: {len(sanitized_messages)}")
        print(f"Messages after truncation: {len(truncated_messages)}")

        # This is the problematic state - orphans exist after our current processing
        # The Anthropic API would reject this request
        if orphan_count > 0:
            print(
                "[CONFIRMED] Bug reproduced: truncation created orphaned tool_result blocks"
            )
            # Mark as expected failure state
            pytest.xfail(
                f"BUG: Current order (sanitize → truncate) creates {orphan_count} orphaned "
                "tool_result blocks. The Anthropic API would reject this with "
                "'unexpected tool_use_id found in tool_result blocks'"
            )

    def test_correct_order_no_orphans(self) -> None:
        """EXPECTED BEHAVIOR: Correct order (truncate → sanitize) leaves no orphans.

        This test shows what should happen:
        1. Create a conversation with many tool_use/tool_result pairs
        2. Apply truncation first (removes old messages, creates orphans)
        3. Apply sanitization second (removes the orphans)
        4. VERIFY: No orphaned blocks remain (CORRECT!)
        """
        # Create conversation with 10 tool call pairs
        messages = self._create_conversation_with_tool_calls(num_pairs=10)

        # Verify no orphans initially
        assert self._count_orphaned_tool_results(messages) == 0
        assert self._count_orphaned_tool_uses(messages) == 0

        # STEP 1: Truncation FIRST (correct order)
        request = {
            "model": "claude-3-opus-20240229",
            "messages": messages,
        }

        truncated_request, was_truncated = truncate_to_fit(
            request,
            max_input_tokens=2000,
            preserve_recent=4,
            safety_margin=0.9,
        )

        assert was_truncated, "Should have been truncated"
        truncated_messages = truncated_request["messages"]

        # After truncation, we might have orphans
        orphans_after_truncation = self._count_orphaned_tool_results(truncated_messages)
        print(f"\nOrphaned tool_results after truncation: {orphans_after_truncation}")

        # STEP 2: Sanitization SECOND (correct order)
        final_messages = _sanitize_tool_results(truncated_messages)

        # After sanitization, no orphans should remain
        final_orphan_count = self._count_orphaned_tool_results(final_messages)

        print(f"Orphaned tool_results after sanitization: {final_orphan_count}")

        assert final_orphan_count == 0, (
            f"After correct order (truncate → sanitize), should have 0 orphans "
            f"but found {final_orphan_count}"
        )

    def test_superdesign_scenario_simulation(self) -> None:
        """Simulate the SuperDesign scenario with massive tool calls.

        SuperDesign (VS Code extension using @ai-sdk/anthropic) sends native
        Anthropic format requests. When a conversation has many tool calls,
        the context exceeds Claude's limit. The client (or proxy) compacts
        the conversation, which can leave orphaned tool blocks.

        This test simulates that scenario.
        """
        # SuperDesign typically has MANY tool calls in a single session
        # Simulate a session with 30 tool call pairs (like reading/writing many files)
        messages: list[Message] = []
        messages.append(
            {
                "role": "user",
                "content": "Implement a new feature across multiple files.",
            }
        )

        # Simulate 30 tool calls (read/write operations)
        for i in range(30):
            tool_id = f"toolu_{i:05d}"
            # Large tool input to simulate real file operations
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": tool_id,
                            "name": "Write" if i % 2 else "Read",
                            "input": {
                                "file_path": f"/project/src/component_{i}.tsx",
                                "content": "Large file content... " * 100
                                if i % 2
                                else None,
                            },
                        }
                    ],
                }
            )
            # Tool result with file contents
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_id,
                            "content": f"{'File written successfully' if i % 2 else 'import React...' * 50}",
                        }
                    ],
                }
            )

        # Final user message
        messages.append({"role": "user", "content": "Great, now run the tests."})

        print("\n[SuperDesign Simulation]")
        print(f"Total messages: {len(messages)}")
        print("Tool call pairs: 30")

        # Current (buggy) flow: sanitize → truncate
        request = {"model": "claude-sonnet-4-20250514", "messages": messages}

        # Step 1: Sanitize first (no orphans yet)
        sanitized = _sanitize_tool_results(messages)

        # Step 2: Truncate (creates orphans!)
        truncated_request, was_truncated = truncate_to_fit(
            {"model": request["model"], "messages": sanitized},
            max_input_tokens=50000,  # Realistic limit
            preserve_recent=10,
            safety_margin=0.9,
        )

        if was_truncated:
            truncated_messages = truncated_request["messages"]
            orphan_results = self._count_orphaned_tool_results(truncated_messages)
            orphan_uses = self._count_orphaned_tool_uses(truncated_messages)

            print(f"Messages after truncation: {len(truncated_messages)}")
            print(f"Orphaned tool_results: {orphan_results}")
            print(f"Orphaned tool_uses: {orphan_uses}")

            if orphan_results > 0 or orphan_uses > 0:
                print(
                    "[CONFIRMED] SuperDesign scenario reproduced - would fail at Anthropic API"
                )
                pytest.xfail(
                    f"BUG: SuperDesign scenario creates {orphan_results} orphaned tool_results "
                    f"and {orphan_uses} orphaned tool_uses after truncation. "
                    "Anthropic API would reject this request."
                )
        else:
            print("Truncation not needed for this token limit")

    def test_full_pipeline_current_vs_fixed(self) -> None:
        """Compare current (buggy) pipeline vs fixed pipeline.

        This test explicitly shows both pipelines side by side.
        """
        messages = self._create_conversation_with_tool_calls(num_pairs=15)

        # === CURRENT (BUGGY) PIPELINE ===
        # This is what the adapters currently do
        current_messages = list(messages)  # Copy

        # Step 1: Sanitize
        current_messages = _sanitize_tool_results(current_messages)

        # Step 2: Truncate
        current_request, _ = truncate_to_fit(
            {"model": "claude-3-opus-20240229", "messages": current_messages},
            max_input_tokens=3000,
            preserve_recent=4,
            safety_margin=0.9,
        )
        current_final = current_request["messages"]
        current_orphans = self._count_orphaned_tool_results(current_final)

        # === FIXED PIPELINE ===
        # This is what should happen
        fixed_messages = list(messages)  # Copy

        # Step 1: Truncate FIRST
        fixed_request, _ = truncate_to_fit(
            {"model": "claude-3-opus-20240229", "messages": fixed_messages},
            max_input_tokens=3000,
            preserve_recent=4,
            safety_margin=0.9,
        )
        fixed_messages = fixed_request["messages"]

        # Step 2: Sanitize SECOND
        fixed_final = _sanitize_tool_results(fixed_messages)
        fixed_orphans = self._count_orphaned_tool_results(fixed_final)

        print("\n[Pipeline Comparison]")
        print(f"Current pipeline (sanitize→truncate): {current_orphans} orphans")
        print(f"Fixed pipeline (truncate→sanitize): {fixed_orphans} orphans")

        # Current pipeline has orphans (bug)
        # Fixed pipeline has no orphans (correct)
        assert fixed_orphans == 0, "Fixed pipeline should have no orphans"

        if current_orphans > 0:
            print(
                f"[BUG CONFIRMED] Current pipeline leaves {current_orphans} orphaned blocks"
            )
            pytest.xfail(
                f"Current pipeline creates {current_orphans} orphans while "
                f"fixed pipeline creates {fixed_orphans}"
            )

    def test_preserve_recent_splits_tool_pair(self) -> None:
        """CRITICAL: Test when preserve_recent boundary splits a tool_use/tool_result pair.

        The bug occurs when:
        1. preserve_recent is set to a value that splits a tool_use (assistant) from its tool_result (user)
        2. The assistant message with tool_use goes into truncatable and gets removed
        3. The user message with tool_result stays in preserved
        4. Result: orphaned tool_result with no matching tool_use

        This is the exact scenario causing "unexpected tool_use_id" errors.
        """
        # DESIGN: Make tool_use the ONLY message in truncatable
        # So ANY truncation will remove it, leaving the tool_result orphaned
        #
        # Structure with preserve_recent=2:
        # [0] assistant: tool_use (LARGE to force truncation) <- In truncatable, REMOVED
        # [1] user: tool_result                               <- In preserved (ORPHAN!)
        # [2] user: final message                             <- In preserved

        messages: list[Message] = [
            # Tool use that will be the ONLY item in truncatable - WILL BE REMOVED
            # Make it large enough to force truncation
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "orphan_tool_123",
                        "name": "analyze_data",
                        "input": {
                            "dataset": "x" * 2000
                        },  # Large input to exceed token limit
                    }
                ],
            },
            # Tool result in preserved - will become orphan when tool_use is removed
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "orphan_tool_123",
                        "content": "Analysis complete: found 42 items",
                    }
                ],
            },
            # Final message in preserved
            {"role": "user", "content": "Thanks, now summarize the results."},
        ]

        print("\n[Targeted Split Test - Verify Fix]")
        print(f"Total messages: {len(messages)}")
        print("Using preserve_recent=2 to keep only last 2 messages")
        print("truncatable will contain ONLY the tool_use message")

        # FIXED ORDER: Truncate FIRST, then Sanitize
        # This is the correct order that prevents orphaned tool_result blocks

        # Step 1: Truncate with preserve_recent=2
        # - preserved = [msg[1], msg[2]] = [tool_result, final message]
        # - truncatable = [msg[0]] = [tool_use ONLY]
        # Truncation will remove the tool_use, potentially creating an orphan
        request = {"model": "claude-3-opus-20240229", "messages": messages}
        truncated_request, was_truncated = truncate_to_fit(
            request,
            max_input_tokens=200,  # Low enough to force removal of the large tool_use
            preserve_recent=2,  # Keep last 2: tool_result + final message
            safety_margin=0.9,
        )

        # Step 2: Sanitize AFTER truncation (the fix)
        # This will clean up any orphaned tool_result blocks created by truncation
        if was_truncated:
            truncated_request["messages"] = _sanitize_tool_results(
                truncated_request["messages"]
            )

        if was_truncated:
            final_messages = truncated_request["messages"]
            print(f"Messages after truncation: {len(final_messages)}")

            # Check what we have
            for i, msg in enumerate(final_messages):
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, list):
                    types = [b.get("type") for b in content if isinstance(b, dict)]
                    ids = [
                        b.get("id") or b.get("tool_use_id")
                        for b in content
                        if isinstance(b, dict)
                    ]
                    print(f"  [{i}] {role}: {types} {ids}")
                else:
                    content_preview = str(content)[:60] if content else "(empty)"
                    print(f"  [{i}] {role}: {content_preview}...")

            orphan_count = self._count_orphaned_tool_results(final_messages)
            print(f"Orphaned tool_results: {orphan_count}")

            # Check if we have tool_result without tool_use
            has_tool_result = any(
                isinstance(msg.get("content"), list)
                and any(
                    b.get("type") == "tool_result"
                    for b in msg["content"]
                    if isinstance(b, dict)
                )
                for msg in final_messages
            )
            has_tool_use = any(
                isinstance(msg.get("content"), list)
                and any(
                    b.get("type") == "tool_use"
                    for b in msg["content"]
                    if isinstance(b, dict)
                )
                for msg in final_messages
            )

            print(f"Has tool_result: {has_tool_result}, Has tool_use: {has_tool_use}")

            # After fix: truncation followed by sanitization should leave no orphans
            # This test now verifies the CORRECT behavior
            if orphan_count > 0:
                # If orphans exist, the fix is not working properly
                pytest.fail(
                    f"REGRESSION: truncation created {orphan_count} orphaned tool_result(s). "
                    "Sanitization should run AFTER truncation to clean up orphans."
                )

            # Verify that we have tool_result but no tool_use (expected after truncation)
            # This is fine as long as there are no ORPHANED tool_results
            if has_tool_result and not has_tool_use:
                print(
                    "[CORRECT] Tool_use was truncated, tool_result was kept but sanitized"
                )
                print(
                    "This is the expected behavior when sanitization runs AFTER truncation"
                )

            print(
                "[FIX VERIFIED] No orphaned tool_result blocks after truncation + sanitization"
            )
        else:
            print("No truncation occurred - need lower token limit")

    def test_token_removal_splits_tool_pair(self) -> None:
        """Test when token-based removal stops in the middle of a tool pair.

        Even if preserve_recent doesn't split a pair, the token-based removal might:
        1. Start removing oldest messages
        2. Remove the assistant message with tool_use
        3. Stop (under token limit) before removing the user message with tool_result
        4. Result: orphaned tool_result
        """
        # Create messages where removal is likely to stop mid-pair
        messages: list[Message] = [
            {"role": "user", "content": "Start"},  # Small - will be removed
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_X",
                        "name": "analyze",
                        "input": {"data": "x" * 500},
                    }
                ],
            },  # Medium - might be removed
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_X",
                        "content": "Result X",
                    }
                ],
            },  # Small - might NOT be removed (leaving orphan)
            {"role": "user", "content": "Middle message"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_Y",
                        "name": "process",
                        "input": {"data": "y"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_Y",
                        "content": "Result Y",
                    }
                ],
            },
            {"role": "user", "content": "Final message with important context"},
        ]

        print("\n[Token Removal Split Test]")

        # Current (buggy) pipeline
        sanitized = _sanitize_tool_results(messages)

        # Use preserve_recent that includes all tool_Y messages but not tool_X
        request = {"model": "claude-3-opus-20240229", "messages": sanitized}
        truncated_request, was_truncated = truncate_to_fit(
            request,
            max_input_tokens=800,
            preserve_recent=4,  # Keep last 4 messages (includes tool_Y pair + final)
            safety_margin=0.9,
        )

        if was_truncated:
            final_messages = truncated_request["messages"]
            orphan_count = self._count_orphaned_tool_results(final_messages)

            print(f"Messages after truncation: {len(final_messages)}")
            print(f"Orphaned tool_results: {orphan_count}")

            if orphan_count > 0:
                print("[BUG CONFIRMED] Token removal split a tool pair")
                pytest.xfail(f"Token removal created {orphan_count} orphan(s)")

    def test_few_messages_massive_content(self) -> None:
        """Test truncation when there are fewer messages than preserve_recent but massive content.

        This reproduces the real-world issue where:
        - Only 9 messages exist
        - preserve_recent=10 (more than message count)
        - Total tokens ~500k (way over 200k limit)
        - OLD BEHAVIOR: truncate_to_fit gave up and returned unchanged
        - NEW BEHAVIOR: reduce preserve_recent dynamically & truncate content blocks

        This was causing "prompt is too long: 200661 tokens > 200000 maximum" errors.
        """
        # Create a few messages with MASSIVE content (simulating large tool results)
        messages: list[Message] = [
            {"role": "user", "content": "Read the README file"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_readme",
                        "name": "Read",
                        "input": {"file_path": "/project/README.md"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_readme",
                        # Massive content - simulate 100k+ chars (like a big file)
                        "content": "README content line\n" * 20000,  # ~400k chars
                    }
                ],
            },
            {"role": "user", "content": "Now read the package.json"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "tool_package",
                        "name": "Read",
                        "input": {"file_path": "/project/package.json"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_package",
                        # Another massive tool result
                        "content": "package.json content line\n" * 15000,  # ~375k chars
                    }
                ],
            },
            {"role": "user", "content": "Summarize both files for me."},
        ]

        print("\n[Few Messages, Massive Content Test]")
        print(f"Total messages: {len(messages)}")
        print("preserve_recent: 10 (more than message count)")

        # OLD behavior would give up here since len(messages)=7 < preserve_recent=10
        request = {"model": "claude-3-opus-20240229", "messages": messages}
        truncated_request, was_truncated = truncate_to_fit(
            request,
            max_input_tokens=50000,  # Target ~45k tokens
            preserve_recent=10,  # More than messages count (7)
            safety_margin=0.9,
        )

        # NEW behavior should:
        # 1. Reduce preserve_recent dynamically
        # 2. Truncate large content blocks as fallback

        print(f"Was truncated: {was_truncated}")

        if was_truncated:
            final_messages = truncated_request["messages"]
            print(f"Messages after truncation: {len(final_messages)}")

            # Check total content size after truncation
            total_chars = 0
            for msg in final_messages:
                content = msg.get("content")
                if isinstance(content, str):
                    total_chars += len(content)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "tool_result":
                                total_chars += len(str(block.get("content", "")))
                            elif block.get("type") == "text":
                                total_chars += len(str(block.get("text", "")))

            print(f"Total content chars after truncation: {total_chars}")

            # Verify content was actually reduced
            # Original was ~775k chars, should be significantly less now
            assert total_chars < 100000, (
                f"Content should be significantly reduced (was {total_chars} chars)"
            )

            # Also verify no orphans were created
            orphan_count = self._count_orphaned_tool_results(final_messages)
            print(f"Orphaned tool_results: {orphan_count}")

            # Run sanitization as would happen in real pipeline
            from ccproxy.llms.formatters.openai_to_anthropic.requests import (
                _sanitize_tool_results,
            )

            sanitized = _sanitize_tool_results(final_messages)
            final_orphan_count = self._count_orphaned_tool_results(sanitized)

            assert final_orphan_count == 0, (
                f"After sanitization should have no orphans but found {final_orphan_count}"
            )

            print("[SUCCESS] Truncation handled few messages with massive content")
        else:
            # If not truncated, verify we're under the limit
            from ccproxy.llms.utils.token_estimation import estimate_request_tokens

            tokens = estimate_request_tokens(request)
            print(f"Request tokens: {tokens}")
            print("[INFO] No truncation needed - content under limit")
