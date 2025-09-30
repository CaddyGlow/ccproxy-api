# Formatter Refactor Plan

## Catalog of Current Entry Points

### `ccproxy/llms/formatters/openai_to_openai/`
- `convert__openai_responses_usage_to_openai_completion__usage` / `convert__openai_completion_usage_to_openai_responses__usage` (lines 732, 767) move token usage data between Responses and Chat envelopes, feeding stream terminators and usage metrics.
- `convert__openai_responses_to_openaichat__request` / `convert__openai_chat_to_openai_responses__request` (lines 797, 3217) translate request bodies, persist tool metadata via `register_request_tools`, and cache instruction text with `register_request` while deriving reasoning defaults from environment toggles.
- `convert__openai_chat_to_openai_responses__response` / `convert__openai_responses_to_openai_chat__response` (lines 1000, 1147) rebuild assistant payloads, merge `<thinking>` XML segments, reconstruct tool calls, and invoke usage helpers such as `_extract_reasoning_blocks` and `_convert_tools_*`.
- `convert__openai_responses_to_openai_chat__stream` / `convert__openai_chat_to_openai_responses__stream` (lines 1257, 1954) orchestrate the Responses↔Chat streaming conversions, maintaining large mutable state machines that manage sequence counters, reasoning buffers, tool-call arguments, obfuscation tokens, and usage snapshots via `OpenAIAccumulator`.

### `ccproxy/llms/formatters/anthropic_to_openai/`
- `convert__anthropic_usage_to_openai_completion__usage` / `convert__anthropic_usage_to_openai_responses__usage` (lines 49, 75) normalize Claude usage objects with `anthropic_usage_snapshot` for both OpenAI surfaces.
- `convert__anthropic_to_openai__error` (line 97) maps Anthropic error envelopes onto OpenAI `ErrorResponse` using constants from `constants.py`.
- `convert__anthropic_message_to_openai_responses__stream` (line 152) is a 1k+ line generator handling Claude streaming with `ClaudeAccumulator`, inserting `<thinking>` tags, emitting tool deltas, tracking usage, and building obfuscation tokens.
- `convert__anthropic_message_to_openai_responses__request` / `response` / `chat__request` / `chat__response` (lines 1445, 1731, 1796, 2010) reshape Anthropic message payloads for OpenAI Responses and Chat APIs, caching instructions and delegating tool-schema conversion to `_build_responses_payload_from_anthropic_request`.
- `convert__anthropic_message_to_openai_chat__stream` (line 1543) replays Claude streams as ChatCompletion chunks, juggling accumulator lookups, finish reasons, usage aggregation, and reasoning markers.

### `ccproxy/llms/formatters/openai_to_anthropic/`
- `convert__openai_to_anthropic__error` (line 23) wraps OpenAI error payloads in Anthropic envelopes using shared mappings.
- `convert__openai_responses_usage_to_openai_completion__usage` / `convert__openai_responses_usage_to_anthropic__usage` (lines 193, 217) bridge OpenAI usage data into both OpenAI-legacy and Anthropic models.
- `convert__openai_chat_to_anthropic_message__request` / `convert__openai_responses_to_anthropic_message__request` (lines 230, 532) collapse heterogeneous OpenAI request formats into Anthropic CreateMessage requests, calling `derive_thinking_config` (line 761).
- `convert__openai_responses_to_anthropic_message__response` / `convert__openai_chat_to_anthropic_messages__response` (lines 798, 1365) rebuild Anthropic message responses, parsing `<thinking>` tags, image blocks, and tool calls.
- `convert__openai_responses_to_anthropic_messages__stream` / `convert__openai_chat_to_anthropic_messages__stream` (lines 950, 1205) translate OpenAI streaming deltas into Anthropic message events, reconstituting content/tool blocks, managing usage metadata, and parsing function-call arguments.

### Shared Glue
- `ccproxy/llms/formatters/utils.py` centralizes usage snapshots, finish-reason mapping, argument parsing, and obfuscation token helpers, yet identifier normalization and reasoning parsing remain duplicated in formatter modules.
- `ccproxy/llms/formatters/context.py` exposes `register_request`, `get_last_request`, and `get_last_instructions`, supplying request-context state for streaming conversions.
- `ccproxy/llms/formatters/constants.py` hosts cross-provider finish/error mappings and default token limits but does not cover the duplicated regex patterns and identifier utilities.
- Tests touch the public converters via `tests/unit/llms/test_openai_to_openai_reasoning.py`, `tests/unit/llms/adapters/test_anthropic_to_openai_helpers.py`, `tests/unit/llms/test_formatter_endpoint_samples.py`, and `tests/integration/test_streaming_converters.py` to validate end-to-end parity.

## Cross-Cutting Pain Points
- Monolithic modules (~2–3k LOC each) mix request shaping, response assembly, streaming state machines, and helper utilities, making change impact and review difficult.
- Critical primitives (identifier normalization, `<thinking>` parsing, tool-choice coercion) are reimplemented across modules, creating drift risk when vendor specs evolve.
- Streaming converters rely on sprawling mutable dicts and nested closures, which are hard to unit test; invariants such as sequence counters, reasoning buffers, and obfuscation tokens are implicit.
- Context propagation depends on module-level globals (`_LAST_REQUEST_TOOLS`, contextvars) with side effects spread across code paths, coupling unrelated conversions.
- Existing tests skew integration-heavy; we lack focused coverage for potential extracted helpers or state-machine behaviors, increasing regression risk during refactors.

## Refactor Plan
1. **Stabilize Scope & Contracts**
   - Draft an architectural brief cataloging each public converter, required schema parity, environment toggles, and context side effects; review against `CONVENTIONS.md` and `TESTING.md` to lock external expectations before touching code.
   - Deliverable: checklist documenting invariants and compatibility requirements for maintainers.

2. **Extract Shared Primitives**
   - Introduce `ccproxy/llms/formatters/common/` for shared helpers (identifier normalization, thinking regex parsing, tool-choice mapping, reasoning segment merging) currently duplicated across modules.
   - Add targeted unit tests under `tests/unit/llms/common/` to pin helper semantics prior to modifying higher-level converters.

3. **Modularize Streaming State Machines**
   - Encapsulate streaming logic in adapter classes (e.g., `OpenAIResponsesToChatStreamAdapter`) that own mutable state and expose `iter_events(...)` methods, replacing the current nested generator closures.
   - Create focused unit tests with synthetic stream fixtures to validate sequencing, tool-call deltas, reasoning summaries, usage aggregation, and obfuscation tokens independent of the full modules.

4. **Split Request/Response Facades**
   - Reorganize each provider-pair package into `requests.py`, `responses.py`, `streams.py`, and `errors.py`, re-exporting existing function names from a façade module to avoid churn.
   - Update imports and maintain existing public API surface; rerun `tests/unit/llms/test_formatter_endpoint_samples.py` to ensure coverage continues to exercise the public functions.

5. **Consolidate Usage & Context Handling**
   - Harmonize usage converters to rely on `UsageSnapshot` helpers exclusively and consider relocating them into `common/usage.py` for clarity.
   - Replace module-level globals for request/tool caching with explicit context objects carried through adapters or stored via `StreamingConfigurable`, reducing hidden dependencies.

6. **Documentation & Validation**
   - Produce contributor-facing documentation (e.g., `docs/development/formatter-guide.md`) summarizing the new structure, helper catalog, and extension guidance; confirm agent-facing behavior remains compliant with `CLAUDE.md`.
   - Run `make check` and `make test-unit` after each milestone; schedule targeted integration runs (`make test-integration PLUGIN=<name>`) before merging structural changes.
