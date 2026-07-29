#!/usr/bin/env bash
# Validate Out of Context on the target board (e.g. Orange Pi 2W).
# Measures raw generation speed, peak memory, and the overflow crash.
#
# Usage: scripts/check-device.sh [path-to-binary] [path-to-model.gguf]
# Defaults assume you run it from the repo root after copying a model into ./models.

set -u

BIN="${1:-./out-of-context}"
MODEL="${2:-models/Bonsai-4B-Q1_0.gguf}"
THREADS="${THREADS:-4}"

if [ ! -x "$BIN" ]; then echo "binary not found/executable: $BIN" >&2; exit 1; fi

echo "Binary : $BIN"
echo "Model  : $MODEL"
echo "Threads: $THREADS"
echo

run() { # max_tokens -> seconds (wall), output discarded
  local n="$1" start end
  start=$(date +%s.%N)
  "$BIN" --model "$MODEL" --threads "$THREADS" --words-per-second 0 \
         --seed 1 --max-tokens "$n" --quiet >/dev/null 2>&1
  end=$(date +%s.%N)
  awk -v s="$start" -v e="$end" 'BEGIN{printf "%.3f", e-s}'
}

echo "== Speed (delta of 200 vs 40 tokens cancels model-load time) =="
t200=$(run 200); t40=$(run 40)
toks_per_sec=$(awk -v a="$t200" -v b="$t40" 'BEGIN{d=a-b; if(d<=0){print "n/a"} else printf "%.2f", 160/d}')
echo "  200 tok: ${t200}s   40 tok: ${t40}s"
echo "  raw speed ~ ${toks_per_sec} tok/s  (~1.4 tokens per word; target >= ~2.1 tok/s for 1.5 words/sec)"
echo

echo "== Peak memory (VmHWM while generating, ctx 512) =="
"$BIN" --model "$MODEL" --threads "$THREADS" --words-per-second 0 \
       --seed 1 --max-tokens 300 --context-size 512 --quiet >/dev/null 2>&1 &
pid=$!; max=0
while kill -0 "$pid" 2>/dev/null; do
  h=$(grep VmHWM "/proc/$pid/status" 2>/dev/null | grep -oE '[0-9]+')
  [ -n "$h" ] && [ "$h" -gt "$max" ] && max=$h
  sleep 0.1
done
echo "  peak RSS ~ $((max/1024)) MB  (board has 1.5GB = ~1536MB; leave room for the OS)"
echo

echo "== Crash test (runs to context overflow at a small context) =="
"$BIN" --model "$MODEL" --threads "$THREADS" --words-per-second 0 \
       --seed 1 --context-size 256 --quiet >/dev/null 2>/tmp/ooc-crash.txt
if grep -q "Context overflow" /tmp/ooc-crash.txt; then
  echo "  OK: reached context overflow and panicked as intended."
else
  echo "  NOTE: ended without the overflow panic. stderr tail:"; tail -3 /tmp/ooc-crash.txt
fi
echo
echo "Done. If speed is below ~1.5 words/sec or RSS is too tight, try a lighter model:"
echo "  --model models/SmolLM2-360M-Instruct-Q4_K_M.gguf"
