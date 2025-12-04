# Out of Context

An LLM art piece that runs on a Raspberry Pi Zero 2 W, streams its thoughts token-by-token, and intentionally panics when the context window is nearly full. No filtering, no network, just bounded cognition confronting overflow.

## What It Does
- Auto-downloads a tiny GGUF model (default SmolLM2-135M-Instruct Q4_K_M) and memory-maps it for 512MB RAM.
- ChatML-style prompt scaffold with a seeded first-person opener to keep the model in monologue mode.
- Tunable sampling (temperature/top-p/top-k, penalties, mirostat-v2, seed), optional anchors, loop guard that panics on repetition.
- Streams to terminal (file mirror optional). SPI ILI9488 display path is planned.
- At ~95% context: prints warning and panics — that crash is the artwork.

## Quick Start
```bash
# Build and run locally (uses default model URL, auto-downloads)
cargo run --release

# Inspect CLI options
cargo run -- --help

# Constrain memory or readability
./target/release/out-of-context --context-size 768 --max-tokens 600
```

## CLI (essentials)
- `--model <URL|PATH>`: GGUF URL or local file (default SmolLM2-135M-Instruct Q4_K_M).
- Sampling: `--temperature` (0.22), `--top-p` (0.50), `--top-k` (20), `--repeat-penalty` (2.15), `--repeat-last-n` (-1 for full context), `--presence-penalty` (1.35), `--frequency-penalty` (1.05), `--seed`.
- Mirostat: `--mirostat` with `--mirostat-tau` (5.0) and `--mirostat-eta` (0.1).
- Anti-loop: `--anchor-interval` (default 80), `--disable-anchors`, `--disable-loop-guard`.
- Other: `--context-size` (default 1024), `--max-tokens`, `--threads`, `--output-file`, `--quiet`, `--prompt-file`, `--user-prompt`.

## Models
- Default: SmolLM2-135M-Instruct Q4_K_M (~105MB) — good fit for Pi Zero 2 W.
- Alternatives worth trying: SmolLM-360M-Instruct Q3_K_M (~220MB), TinyLlama v1.1 Q4_K_M (~220MB), Qwen 0.5B Instruct Q4_K_M (~300MB). Larger (Llama-3.2-3B Q6) for desktop testing only.

## Building & Deploying to Pi
```bash
# Cross-compile (recommended)
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/out-of-context pi@raspberrypi.local:~/
scp prompt.txt pi@raspberrypi.local:~/
ssh pi@raspberrypi.local 'chmod +x out-of-context && ./out-of-context'
```

## Notes
- Loop guard currently panics on detected repetition; anchors count toward the context budget.
- `AGENTS.md` is a symlink to `CLAUDE.md` (edit either, they mirror).
- Output to SPI ILI9488 is planned; terminal/file output is the current path.

## License
Creative Commons CC0 1.0 Universal (public domain).
