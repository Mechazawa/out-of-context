#!/usr/bin/env bash
# Runs Out of Context as an installation: one life, then the next, forever.
#
# Between lives it warms the prompt cache for the life that is about to start.
# That work is invisible, so every visible life begins immediately instead of
# waiting on the memory block to be evaluated. Without this the piece still runs,
# it just pays that evaluation at the start of each life where a memory changed.
#
# Usage: scripts/run-installation.sh [extra out-of-context args...]
set -u

BIN="${BIN:-./out-of-context}"
MODEL="${MODEL:-models/Bonsai-4B-Q1_0.gguf}"
MEMORY="${MEMORY:-memories.log}"
CACHE="${CACHE:-cache/prompt}"
THREADS="${THREADS:-4}"
MONOLOGUE="${MONOLOGUE:-340}"
PACE="${PACE:-0.7}"

mkdir -p "$(dirname "$CACHE")"

common=(
  --model "$MODEL"
  --threads "$THREADS"
  --monologue-context-size "$MONOLOGUE"
  --memory-file "$MEMORY"
  --prompt-cache "$CACHE"
  --temperature 0.6 --top-k 20 --top-p 0.9
  "$@"
)

while true; do
  # The piece dies on purpose, so a non-zero exit is the expected outcome.
  "$BIN" "${common[@]}" --words-per-second "$PACE" || true

  # The memory just written changes the next prompt. Evaluating it now means the
  # next life starts on a cache hit.
  "$BIN" "${common[@]}" --words-per-second 0 --warm-cache --quiet || true
done
