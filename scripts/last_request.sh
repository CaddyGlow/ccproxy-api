#!/usr/bin/env bash
# Use a function - it won't visually expand
fdcat() {
  fd -t f "$@" -x sh -c 'printf "\n=== %s ===\n" "$1" && cat "$1"' _ {}
}
PATH_LOG="/tmp/ccproxy"
PATH_REQ="${PATH_LOG}/tracer/"
COMMAND_REQ="${PATH_LOG}/command_replay"

# Parse arguments
N=-1 # Default to last request
REQUEST_ID=""
if [[ $# -gt 0 ]]; then
  if [[ $1 =~ ^-[0-9]+$ ]]; then
    N=$1
  elif [[ $1 =~ ^[a-f0-9]{8}$ ]] || [[ $1 =~ ^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$ ]]; then
    REQUEST_ID=$1
  else
    echo "Usage: $0 [-N|request_id]"
    echo "  -N: Show the Nth-to-last request (e.g., -1 for last, -2 for second-to-last)"
    echo "  request_id: Show the request with the given 8-char hex ID or full UUID"
    exit 1
  fi
fi

if [[ -n "$REQUEST_ID" ]]; then
  LAST_UUID="$REQUEST_ID"
else
  # Get the Nth-to-last ID (grouped by unique ID, preserving chronological order)
  # Handle both 8-char hex IDs and full UUIDs
  # Extract IDs from filenames, prioritizing the file modification order
  ALL_IDS=$(eza -la --sort=modified "${PATH_REQ}" | sed -n -E '
    s/^.*[[:space:]]([a-f0-9]{8})_[0-9]{8}_[0-9]{6}_[0-9]{6}_[0-9]{6}_.*\..*$/\1/p
    s/^.*[[:space:]]([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})_[0-9]{8}_[0-9]{6}_[0-9]{6}_[0-9]{6}_.*\..*$/\1/p
  ')
  UNIQUE_IDS=$(echo "$ALL_IDS" | awk '{if(!seen[$0]++) print}')

  if [[ $N == -1 ]]; then
    LAST_UUID=$(echo "$UNIQUE_IDS" | tail -1)
  else
    # Convert negative index to positive from end: -2 becomes 2nd from end, -3 becomes 3rd from end
    POS_FROM_END=$((${N#-}))
    LAST_UUID=$(echo "$UNIQUE_IDS" | tail -n "$POS_FROM_END" | head -1)
  fi
fi

if [[ -z "$LAST_UUID" ]]; then
  if [[ -n "$REQUEST_ID" ]]; then
    echo "No request found for ID $REQUEST_ID"
  else
    echo "No request found for position $N"
  fi
  exit 1
fi

printf "\n## Log\n"
printf '```json\n'
rg -I -t log "${LAST_UUID}" ${PATH_LOG} | jq .
printf '```\n'

printf "\n## Raw\n"
for f in "${PATH_REQ}/"*"${LAST_UUID}"*.http; do
  [ -e "$f" ] || continue
  echo "$f"
  printf '```http\n'
  cat "$f"
  printf '```\n'
done

printf "\n## Requests\n"
for f in "${PATH_REQ}/"*"${LAST_UUID}"*.json; do
  [ -e "$f" ] || continue
  echo "$f"
  printf '```json\n'
  cat "$f" | jq .
  printf '```\n'
done

printf "\n## Response Stream\n"
STREAM_FOUND=false
for f in "${PATH_REQ}/"*"${LAST_UUID}"*_streaming_response.json; do
  [ -e "$f" ] || continue
  STREAM_FOUND=true
  echo "$f"
  printf '```json\n'
  jq '{request_id, provider, method, url, total_chunks, total_bytes, buffered_mode}' "$f"
  printf '```\n'
  UPSTREAM_STREAM=$(jq -r '.upstream_stream_text // empty' "$f")
  if [[ -n "$UPSTREAM_STREAM" ]]; then
    printf 'Upstream Stream (provider raw)\n'
    printf '```text\n'
    printf '%s\n' "$UPSTREAM_STREAM"
    printf '```\n'
  fi
  printf 'Client Stream (proxied)\n'
  printf '```text\n'
  jq -r '.response_text' "$f"
  printf '\n```\n'
done

if [[ "$STREAM_FOUND" == false ]]; then
  for f in "${PATH_REQ}/"*"${LAST_UUID}"*response_core_http.json; do
    [ -e "$f" ] || continue
    echo "$f"
    printf '```json\n'
    jq -r .body "$f" | grep '^data: ' | sed 's/^data: //' | jq -r .
    printf '```\n'
  done
fi

printf "\n## Command\n"
for f in "${COMMAND_REQ}/"*"${LAST_UUID}"*.txt; do
  [ -e "$f" ] || continue
  echo "$f"
  printf '```sh\n'
  cat "$f"
  printf '```\n'
done
