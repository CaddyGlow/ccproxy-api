#!/usr/bin/env bash

act \
  -P ubuntu-latest=catthehacker/ubuntu:act-latest \
  --reuse \
  --action-cache-path ~/.act \
  "$@"
