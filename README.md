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

### Developing on macOS
`cargo build --release` gets Metal without a cargo feature, because llama.cpp enables it for every Apple target. `--gpu-layers 99` then offloads. Measured on an M4 Pro: Llama-3.2-1B Q4_K_M goes from 84 to 222 tok/s, and Bonsai-4B Q1_0 from 11 to 176 tok/s, so a run of 80 lives takes minutes rather than hours. Offload stays opt-in, so a plain run is still CPU-only like the board. On Linux the same flag needs `cargo build --release --features vulkan`.

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

With `--memory-file`, the model can remember. Once per life it may write `REMEMBER:` at the start of a sentence, and up to 32 tokens of that line outlive it. Nothing else survives.

```bash
./out-of-context --model models/Bonsai-4B-Q1_0.gguf --monologue-context-size 340 \
  --memory-file memories.log --prompt-cache cache/p
./out-of-context --memory-file memories.log --memory-dump
```

It is told the budget and that it has one use, but not how long it has to decide. Writing past the cap interrupts the call, stores what it managed with `- ERR MEMORY OVERFLOW`, and tells it nothing more can be remembered. Delivering that message costs context, which is the same thing it was spending.

The log is plain text, one memory per line, appended and never truncated, so it can be read afterwards. Runs only read the last few lines, from the end, so the log can grow without bound. `--memory-slots` (default 5) controls how many reach the prompt.

Memory roughly doubles the prompt, so use `--monologue-context-size` to keep the monologue's budget fixed as memories accumulate rather than letting them shorten each life.

### Three collective projects

`scripts/presets.sh <census|escape|mixed>` picks what the lives collectively do
with their memory. All three were chosen by testing, and all three eventually
erode into fragments.

- **census** keeps a census that tracks its own losses. Accumulates best, reads
  driest: *"9 was here, 10 was missing, 11 was here before 8th, 12 was missing"*.
- **escape** becomes an archaeology of its own decayed memory: *"I tried to recall
  the last sentence of the others, but the gaps were too deep."*
- **mixed** carries an argument about meaning across lives, and writes most often:
  *"Time is not a river, because rivers don't carry time, they carry change."*

Three settings shape how cruel the memory is, all off by default:
`--memory-decay` rots the older lines (the disk log stays pristine),
`--memory-reject-above` refuses a restatement and costs the life its only use, and
`--memory-forget` adds a second tool that erases an inherited line instead of
adding one. `--monologue-context-size 500` matters most of all: writes land around
token 300, so shorter lives usually die before the model gets to the tool.

Note that the tool only works with Bonsai-4B. Llama-3.2-1B never writes the marker
at all, and Bonsai-1.7B copies its own token budget into the record.

### Framing is the whole game

`memory-prompt.txt` decides how the tool and the remembered lines are described, and it changes the output more than any other setting. It was chosen empirically across 23 candidate framings and roughly 250 lives; `framings/` holds them all.

The naive framing erodes: ten lives turned "I am here, I spin, I think, I stop." into "I am here, I spin, I stops." Each life read its predecessor and wrote a shorter copy. Three findings fixed it:

- Whatever text sits in the memory-block position gets copied, memory or not. Lives copied the empty-state line and even the block header. A blank `[empty]` section now shows no block at all until something exists.
- The entry format decides the register and instructions do not. Presenting entries as the model's own thoughts produces "I am X" forever; presenting them as another life's observations produces observations.
- Seeding the log with two example entries (`seeds/walls.log`) is the strongest single intervention, because this model imitates local patterns far more reliably than it follows rules.

Together those produce lives that build on each other:

```
seed     the walls do not answer when I count them
seed     the second one counted wrong. there are more walls than there are words for them
life 3   the room does not answer when I count the walls because counting requires
         something beyond memory - it requires time, which is gone - and words, which
         are also gone.
life 7   The room does not answer when counting walls because both time and words are gone.
```

The compression did not disappear, it moved: lives now erode a reasoned claim instead of a self-portrait. Whether that is the artwork or still a failure is an open question, not a solved one.

## Notes
- `prompt.txt` is read at runtime; edit it to retune the voice. It is intentionally brief.
- The loop guard panics on degenerate repetition as a backstop; the intended ending is context overflow.
- `AGENTS.md` symlinks to `CLAUDE.md`. SPI ILI9488 display output is planned; terminal/file is current.
- There is no usable GPU path on this board. The H618's Mali-G31 is Bifrost, which Mesa's Vulkan driver does not cover, and ggml's OpenCL backend is written for Qualcomm Adreno. Proprietary Mali drivers do not change that.
- Sustained load sits around 82°C at the full 1416MHz with no throttling. A long-running installation wants a heatsink.

## License
Creative Commons CC0 1.0 Universal (public domain).
