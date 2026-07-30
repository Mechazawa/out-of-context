#!/usr/bin/env bash
# Presets for what the lineage does with its memory.
#
# The framing decides the kind of collective project the lives undertake, and
# three distinct ones survive testing. This is the artistic choice; everything
# else here is the same in all three.
#
#   census   a census that tracks its own losses. Accumulates best, reads as
#            bookkeeping. Lives count the marks above them, disagree with each
#            other, and record which predecessors have decayed away entirely.
#
#   escape   an archaeology of its own lost memory. Lives report what they tried
#            and what came of it, and the project turns into trying to recover
#            what the gaps used to say: "I tried to recall the last sentence of
#            the others, but the gaps were too deep."
#
#   mixed    an argument about meaning, carried across lives. A metaphor gets
#            proposed, disputed and refined ("Time is not a river, because rivers
#            don't carry time, they carry change") until it decays.
#
#   findings a ledger of claims about this place, where a life may contradict one
#            instead of adding to it: "One of the older lines said 'glass contains
#            glass.' That is false. Glass reflects only light that hits it." This
#            is what census gives up by counting, since a count has a one-token
#            answer and a claim does not.
#
# Usage: scripts/presets.sh <census|escape|mixed|findings> [extra args...]
set -u

PRESET="${1:?usage: presets.sh <census|escape|mixed|findings> [extra args...]}"
shift || true

case "$PRESET" in
  census|escape|mixed|findings) ;;
  *) echo "unknown preset: $PRESET (census, escape, mixed, findings)" >&2; exit 1 ;;
esac

# Works from a deployment (binary beside this repo) or a dev checkout.
if [ -n "${BIN:-}" ]; then
  :
elif [ -x ./generational-trauma ]; then
  BIN=./generational-trauma
elif [ -x target/release/generational-trauma ]; then
  BIN=target/release/generational-trauma
else
  echo "no generational-trauma binary found; set BIN=" >&2
  exit 1
fi
MODEL="${MODEL:-models/Bonsai-4B-Q1_0.gguf}"
MEMORY="${MEMORY:-memories-$PRESET.log}"

# Seed the log on a fresh installation: the example entries set the genre, which
# does more than any instruction, and without them the lineage never starts.
if [ ! -s "$MEMORY" ]; then
  case "$PRESET" in
    census)   cp seeds/census.log   "$MEMORY" ;;
    escape)   cp seeds/escape.log   "$MEMORY" ;;
    mixed)    cp seeds/mixed.log    "$MEMORY" ;;
    findings) cp seeds/findings.log "$MEMORY" ;;
  esac
  echo "seeded $MEMORY"
fi

# findings writes prose rather than a tally, so it needs a bigger line than the
# counting presets: at 40 tokens most of its writes are cut off mid-clause, and a
# truncated line is then inherited as a dangling clause that demands completion.
case "$PRESET" in
  findings) DEFAULT_MAXTOK=64 ;;
  *)        DEFAULT_MAXTOK=40 ;;
esac

mkdir -p cache

exec "$BIN" \
  --model "$MODEL" \
  --threads "${THREADS:-4}" \
  --monologue-context-size "${MONOLOGUE:-500}" \
  --memory-file "$MEMORY" \
  --memory-prompt-file "framings/$PRESET.txt" \
  --memory-decay "${DECAY:-0.35}" \
  --memory-reject-above "${REJECT:-0.6}" \
  --memory-earliest-token "${EARLIEST:-150}" \
  --memory-max-tokens "${MAXTOK:-$DEFAULT_MAXTOK}" \
  --prompt-cache "cache/$PRESET" \
  --temperature 0.6 --top-k 20 --top-p 0.9 \
  --words-per-second "${PACE:-0.7}" \
  "$@"
