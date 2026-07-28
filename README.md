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

# Skip prompt evaluation on every run after the first
cargo run --release -- --prompt-cache prompt.cache

# Inspect all options
cargo run --release -- --help
```

## Target Hardware
Orange Pi 2W (quad Cortex-A53 at 1416MHz, 1.5GB RAM), headless. Cross-compile and deploy:
```bash
cargo install cross
# -mtune only: the A53 is ARMv8.0, so the instruction set must stay at the
# armv8-a baseline. Do not use RUSTFLAGS -C target-cpu here; llama-cpp-sys-2
# turns that into an invalid -march=cortex-a53.
CFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
CXXFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
  cross build --release --target aarch64-unknown-linux-gnu
scp target/aarch64-unknown-linux-gnu/release/out-of-context prompt.txt user@<host>:~/
ssh user@<host> 'chmod +x out-of-context && ./out-of-context --threads 4 --prompt-cache p.cache'
```
Building requires `cmake` and a C/C++ toolchain for llama.cpp.

## CLI (essentials)
- `--model <URL|PATH>`: GGUF URL or local file (default Llama-3.2-1B-Instruct Q4_K_M).
- `--context-size <N>`: context window (default 512). Smaller = shorter, cleaner life; larger = longer run, more tail drift.
- `--words-per-second <F>`: display pace (default 1.5; 0 = as fast as the model runs).
- `--threads <N>`: use 4 on the Orange Pi.
- Sampling: `--temperature` 0.85, `--top-p` 0.95, `--top-k` 64, `--min-p` 0.05, `--repeat-penalty` 1.1, plus DRY (`--dry-multiplier` 0.8, `--dry-base` 1.75, `--dry-allowed-length` 3).
- `--prompt-cache <PATH>`: save the evaluated prompt and reuse it. Worth it on the board, where prompt evaluation costs about 135 seconds per boot.
- `--memory-file <PATH>`: give the model its one tool (see below). `--memory-max-tokens` (32), `--memory-slots` (5), `--memory-dump` to read the archive.
- `--seed <N>` for a reproducible run, `--output-file <PATH>` to log the raw stream, `--quiet`, `--disable-loop-guard`.

## The One Tool

With `--memory-file`, the model can remember. Once per life it may write a line starting `REMEMBER:` and up to 32 tokens will outlive it. Nothing else survives.

```bash
./out-of-context --model models/Bonsai-4B-Q1_0.gguf --context-size 768 \
  --memory-file memories.txt --prompt-cache b4b.cache
./out-of-context --model models/Bonsai-4B-Q1_0.gguf --memory-file memories.txt --memory-dump
```

It is told the budget and that it has one use, but not how long it has to decide. Writing past the cap interrupts the call, stores what it managed with `- ERR MEMORY OVERFLOW`, and tells it that nothing more can be remembered. Delivering that message costs context, which is the same thing it was spending.

The next life is shown the newest five memories as a lossy store, oldest discarded. Every memory ever written is kept on disk regardless, so the archive can be read afterwards even though the model believes the evicted ones are gone.

Memory costs life: the tool description and the block roughly double the prompt, so use `--context-size 768` or more. In early testing the memories did not accumulate so much as erode, each life compressing its predecessor's sentence a little further.

## Models
Default Llama-3.2-1B-Instruct Q4_K_M (~770MB) gives the best monologue voice per unit of compute on this board. Lighter, faster fallbacks if the device is too slow or tight on memory: SmolLM2-360M-Instruct (~270MB), Qwen2.5-0.5B-Instruct (~400MB). Switch with `--model`; all settings work unchanged.

PrismML's 1-bit **Bonsai** family (Q1_0, Apache 2.0) also runs, and Bonsai-4B produces the strongest interior voice of anything tested: no audience-addressing, no assistant reflexes, and it inhabits the situation instead of restating it. The catch is speed. Measured on the board:

| model | file | peak RSS | speed |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | 808MB | 1318MB (x86) | 0.99 tok/s is the 4B figure; 1B not yet measured on-board |
| Bonsai-4B Q1_0 | 572MB | 700MB | 0.99 tok/s, 0.71 words/sec |
| Bonsai-1.7B Q1_0 | 248MB | 367MB (x86) | not yet measured on-board |

```bash
./out-of-context --model models/Bonsai-4B-Q1_0.gguf \
  --temperature 0.6 --top-k 20 --top-p 0.9 --prompt-cache b4b.cache
```

On a Cortex-A53, generation speed tracks parameter count rather than bits per weight: 1-bit quantization cuts memory, not the multiply-accumulate count, and ARMv8.0 has no `SDOT` instruction. Bonsai's published throughput figures come from GPUs and from phone SoCs with dotprod, so they do not transfer. Q1_0 needs `llama-cpp-2` 0.1.153 or newer.

## Notes
- `prompt.txt` is read at runtime; edit it to retune the voice. It is intentionally brief.
- The loop guard panics on degenerate repetition as a backstop; the intended ending is context overflow.
- `AGENTS.md` symlinks to `CLAUDE.md`. SPI ILI9488 display output is planned; terminal/file is current.
- There is no usable GPU path on this board. The H618's Mali-G31 is Bifrost, which Mesa's Vulkan driver does not cover, and ggml's OpenCL backend is written for Qualcomm Adreno. Proprietary Mali drivers do not change that.
- Sustained load sits around 82°C at the full 1416MHz with no throttling. A long-running installation wants a heatsink.

## License
Creative Commons CC0 1.0 Universal (public domain).
