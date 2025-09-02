#!/usr/bin/env bash
body=$(jq -r '."last-assistant-message"// empty')

if [ -z "$body" ]; then body="pinging"; fi

printf "\e]777;notify;%s;%s\e\\" "codex" "$body"
