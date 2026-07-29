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

# Works from a deployment (binary beside this repo) or a dev checkout.
if [ -n "${BIN:-}" ]; then
  :
elif [ -x ./out-of-context ]; then
  BIN=./out-of-context
elif [ -x target/release/out-of-context ]; then
  BIN=target/release/out-of-context
else
  echo "no out-of-context binary found; set BIN=" >&2
  exit 1
fi
MODEL="${MODEL:-models/Bonsai-4B-Q1_0.gguf}"
MEMORY="${MEMORY:-memories.log}"
CACHE="${CACHE:-cache/prompt}"
THREADS="${THREADS:-4}"
# 500 rather than 340: writes land around token 300, so a shorter life usually
# ends before the model gets to the tool. At 340 one life in three writes
# something; at 500 it is three in four. Costs about three more minutes per life
# on the board and 70MB of KV cache.
MONOLOGUE="${MONOLOGUE:-500}"
PACE="${PACE:-0.7}"
SEED_LOG="${SEED_LOG:-seeds/census.log}"
DECAY="${DECAY:-0.35}"
REJECT="${REJECT:-0.6}"

mkdir -p "$(dirname "$CACHE")"

# Seed the log on a fresh installation. Two example entries set the genre by
# example, which the trials found does more than any instruction: without them
# the first life copies whatever text sits where a memory should be, and the
# lineage never starts counting.
if [ ! -s "$MEMORY" ] && [ -f "$SEED_LOG" ]; then
  cp "$SEED_LOG" "$MEMORY"
  echo "seeded $MEMORY from $SEED_LOG"
fi

common=(
  --model "$MODEL"
  --threads "$THREADS"
  --monologue-context-size "$MONOLOGUE"
  --memory-file "$MEMORY"
  --memory-decay "$DECAY"
  --memory-reject-above "$REJECT"
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
