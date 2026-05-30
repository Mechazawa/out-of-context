# Out of Context

An LLM art piece that runs on a small single-board computer, speaks a continuous first-person stream of consciousness one word at a time, and intentionally panics when its context window fills. No filtering, no network. The crash is the artwork: a bounded mind narrating its way to overflow.

## What It Does
- Auto-downloads a GGUF model (default Llama-3.2-1B-Instruct Q4_K_M) and memory-maps it.
- Wraps a brief, deliberately under-directed prompt around a seeded first-person opener, so the voice is genuine rather than scripted.
- Reveals text at a steady reading pace (default 1.5 words/second) and word-wraps to the terminal.
- Suppresses the model's assistant reflexes with a DRY sampler, targeted logit bans (control tokens, markup), and a short context that ends the run before small models drift.
- At ~95% context: prints a warning and panics. The thought cuts off mid-sentence.

## Quick Start
```bash
# Build and run (auto-downloads the model on first run)
cargo run --release

# See the full output unpaced, capped, for inspection
cargo run --release -- --words-per-second 0 --max-tokens 300

# Inspect all options
cargo run --release -- --help
```

## Target Hardware
Orange Pi 2W (quad Cortex-A53, 1.5GB RAM), headless. Cross-compile and deploy:
```bash
cargo install cross
cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/out-of-context prompt.txt orangepi@<host>:~/
ssh orangepi@<host> 'chmod +x out-of-context && ./out-of-context --threads 4'
```

## CLI (essentials)
- `--model <URL|PATH>`: GGUF URL or local file (default Llama-3.2-1B-Instruct Q4_K_M).
- `--context-size <N>`: context window (default 512). Smaller = shorter, cleaner life; larger = longer run, more tail drift.
- `--words-per-second <F>`: display pace (default 1.5; 0 = as fast as the model runs).
- `--threads <N>`: use 4 on the Orange Pi.
- Sampling: `--temperature` 0.85, `--top-p` 0.95, `--top-k` 64, `--min-p` 0.05, `--repeat-penalty` 1.1, plus DRY (`--dry-multiplier` 0.8, `--dry-base` 1.75, `--dry-allowed-length` 3).
- `--seed <N>` for a reproducible run, `--output-file <PATH>` to log the raw stream, `--quiet`, `--disable-loop-guard`.

## Models
Default Llama-3.2-1B-Instruct Q4_K_M (~770MB) gives the best monologue voice on this board. Lighter, faster fallbacks if the device is too slow or tight on memory: SmolLM2-360M-Instruct (~270MB), Qwen2.5-0.5B-Instruct (~400MB). Switch with `--model`; all settings work unchanged.

## Notes
- `prompt.txt` is read at runtime; edit it to retune the voice. It is intentionally brief.
- The loop guard panics on degenerate repetition as a backstop; the intended ending is context overflow.
- `AGENTS.md` symlinks to `CLAUDE.md`. SPI ILI9488 display output is planned; terminal/file is current.
- Raw speed on the real board still needs confirming (target 1 to 2 words/second).

## License
Creative Commons CC0 1.0 Universal (public domain).
