# Formatter Refactor Progress

## Stage 1 – Shared Helpers ✅
- Extracted identifier and reasoning/thinking utilities into `ccproxy/llms/formatters/common/`.
- Updated OpenAI↔OpenAI and cross-provider formatters to use shared helpers.
- Added unit coverage for helper modules (`tests/unit/llms/common/`).

## Stage 2 – Streaming Adapter Encapsulation ✅
- Wrapped OpenAI↔OpenAI streaming conversions in adapter classes.
- Added adapter-level unit smoke tests verifying wrapper delegation.
- Moved Anthropic↔OpenAI and OpenAI↔Anthropic streaming converters into adapter classes with private helper methods.
- End-to-end unit suite (`make test-unit`) and lint/typing (`make check`) passing after each phase.

## Stage 3 – Facade Package Split ✅
- Converted the formatter entry points into package facades (`requests.py`, `responses.py`, `streams.py`, `errors.py`) for OpenAI↔OpenAI, Anthropic↔OpenAI, and OpenAI↔Anthropic.
- Preserved the public API by re-exporting converters and stream adapters while adding module proxies that keep monkey-patching semantics for adapter tests intact.
- Introduced lightweight stream wrapper functions inside the facade modules so adapters remain patchable and kept `make check` and `make test-unit` green after the split.

## Stage 4 – Usage & Context Consolidation ✅
- Centralized token usage translation into `ccproxy/llms/formatters/common/usage.py`, de-duplicating OpenAI↔OpenAI and cross-provider usage mappers.
- Updated formatter modules to delegate to the shared helpers while keeping legacy function names for backward compatibility.
- Moved request tool caching into `ccproxy.llms.formatters.context` so streaming adapters now rely on contextvars instead of module-level globals.

## Stage 5 – Streaming Reasoning Helpers ✅
- Added `ReasoningBuffer` to `ccproxy/llms/formatters/common/streams.py` to encapsulate reasoning text state management for streaming adapters.
- Refactored the OpenAI Responses→Chat streaming adapter to use the shared buffer instead of ad-hoc nested closures, preserving emitting behavior and reasoning XML output.
- Kept streaming tests green (`make check`, `make test-unit`) while simplifying future extraction of additional stream helpers.

## Stage 6 – Streaming Tool State ✅
- Introduced `ToolCallState`, `ToolCallTracker`, and `IndexedToolCallTracker` to manage tool-call lifecycle and output indexing with typed helpers.
- Replaced the remaining dictionary-based tool state in both OpenAI streaming adapters with the new trackers, reducing mutation-heavy logic and clarifying state transitions.
- Centralized obfuscation token generation through `ObfuscationTokenFactory`, keeping deterministic seeds aligned with response identifiers.

## Stage 7 – Anthropic Stream Adoption ✅
- Ported the Anthropic→OpenAI Responses adapter to the shared `IndexedToolCallTracker` and `ToolCallState`, eliminating bespoke dict juggling for tool calls.
- Wired the same adapter into `ObfuscationTokenFactory` so text deltas share the deterministic token strategy used elsewhere.
- Verified the refactor with `make check` and `make test-unit`, ensuring parity across all streaming converters touched so far.

## Stage 8 – OpenAI→Anthropic Streams ✅
- Migrated the OpenAI→Anthropic Responses adapter to `IndexedToolCallTracker`, collapsing its tool argument/meta dicts into the shared state dataclass.
- Preserved Anthropic tool-use content blocks by emitting them directly from the tracker-backed state while keeping accumulator fallbacks intact.
- Confirmed parity via `make check` and `make test-unit`, aligning all three provider pairs on the common streaming scaffolding.

## Stage 9 – OpenAI Chat→Anthropic ✅
- Applied `ToolCallTracker` to the Chat streaming adapter, wiring OpenAI tool-call deltas into the shared state helper and emitting Anthropic `tool_use` blocks from the normalized state.
- Parsed delta tool-call payloads incrementally so partial argument fragments accumulate in the tracker before final emission.
- Verified the migration with `make check` and `make test-unit`, completing the cross-provider adoption of the shared streaming infrastructure.

## Stage 10 – Streaming Utilities ✅
- Added `build_anthropic_tool_use_block` to `common/streams.py`, allowing both OpenAI→Anthropic adapters to emit tool blocks via a shared helper instead of duplicating JSON parsing.
- Audited obfuscation token usage across adapters; only Responses→Chat paths require deterministic tokens, so no additional coverage changes were needed.
- Confirmed helper extraction with `make check` and `make test-unit`.

## Stage 11 – Legacy Modules Removed ✅
- Inlined the Anthropic↔OpenAI, OpenAI↔Anthropic, and OpenAI↔OpenAI implementations into their package facades now that helpers/adapters are stable.
- Deleted the `legacy.py` shims and module-level proxies, so all callers resolve against `requests.py`, `responses.py`, `streams.py`, and `errors.py`.
- Retired the empty top-level formatter modules to avoid import ambiguity, keeping package exports authoritative.

## Next Steps
- Evaluate whether message/content block builders should be shared across adapters for further deduplication.
- Draft contributor documentation covering the new formatter layout and helper catalog before tackling deeper context refactors.
