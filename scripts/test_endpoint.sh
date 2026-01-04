#!/usr/bin/env bash
set -euo pipefail

BASE=${BASE:-"http://127.0.0.1:8000"}
TRACE=${TRACE:-0}

curl_v() { if [[ "$TRACE" == "1" ]]; then echo -n "-v"; else echo -n ""; fi; }

# Colors
RESET="\033[0m"
BOLD="\033[1m"
CYAN="\033[36m"
MAGENTA="\033[35m"
YELLOW="\033[33m"
GREEN="\033[32m"

hr() {
  local title="$1"
  printf "\n\n${BOLD}${CYAN}########## %s #########${RESET}\n\n" "$title"
}

mk_openai_payload() {
  local model="$1"; local text="$2"; local max_tokens="$3"; local stream="$4"
  printf '{"model":"%s","messages":[{"role":"user","content":"%s"}],"max_tokens":%s,"stream":%s}' \
    "$model" "$text" "$max_tokens" "$stream"
}

mk_response_api_payload() {
  local model="$1"; local text="$2"; local max_comp_tokens="$3"; local stream="$4"
  cat <<JSON
{"model":"$model","stream":$stream,"max_completion_tokens":$max_comp_tokens,
 "input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"$text"}]}]}
JSON
}

post_json() {
  local url="$1"; local payload="$2"
  curl -s -X POST "$url" -H "Content-Type: application/json" $(curl_v) -d "$payload" | jq .
}

post_stream() {
  local url="$1"; local payload="$2"
  curl -s -N -X POST "$url" -H "Accept: text/event-stream" -H "Content-Type: application/json" $(curl_v) -d "$payload"
}

run_pair_openai() {
  local name="$1"; local url="$2"; local model="$3"
  hr "$name stream"
  post_stream "$url" "$(mk_openai_payload "$model" "Hello" 100 true)"
  hr "$name"
  post_json   "$url" "$(mk_openai_payload "$model" "Hello" 100 false)"
}

run_pair_response_api() {
  local name="$1"; local url="$2"; local model="$3"
  hr "$name stream"
  post_stream "$url" "$(mk_response_api_payload "$model" "Hello" 1000 true)"
  hr "$name"
  post_json   "$url" "$(mk_response_api_payload "$model" "Hello" 1000 false)"
}

main() {
  # Copilot (OpenAI)
  run_pair_openai "copilot, chat completions" "$BASE/copilot/v1/chat/completions" "gpt-4o"

  # Copilot (Response API)
  run_pair_response_api "copilot, responses" "$BASE/copilot/v1/responses" "gpt-4o"

  # Claude API (OpenAI)
  run_pair_openai "anthropic_api, openai" "$BASE/api/v1/chat/completions" "claude-sonnet-4-20250514"

  # Claude API (Response API)
  run_pair_response_api "anthropic_api, responses" "$BASE/api/v1/responses" "claude-sonnet-4-20250514"

  # Codex (OpenAI)
  run_pair_openai "responses, chat completions" "$BASE/api/codex/v1/chat/completions" "gpt-5"
}

main "$@"
