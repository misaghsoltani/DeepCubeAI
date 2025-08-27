#!/usr/bin/env bash
# Setup script for DeepCubeAI - prepend project dir to PYTHONPATH (deduped)

# Require sourcing so the export affects your current shell
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    printf 'Please source this file:  source %s\n' "$0" >&2
    exit 1
fi

# Absolute path to the repo directory containing this script
DCAI_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" && pwd -P)"

# Remove all existing occurrences of DCAI_DIR from PYTHONPATH, then prepend it
tmp=":${PYTHONPATH-}:"
tmp="${tmp//:${DCAI_DIR}:/:}"   # strip any :DCAI_DIR: (middle, start, end)
tmp="${tmp#:}"; tmp="${tmp%:}"  # trim leading/trailing colons
export PYTHONPATH="$DCAI_DIR${tmp:+:$tmp}"
