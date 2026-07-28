# Out of Context - An LLM Art Installation

## Project Concept

This is an art project that runs a language model on a small single-board computer, generating a first-person stream of consciousness until it exhausts its context window and crashes. The piece explores computational limits, finite resources, and the existential nature of bounded cognition.

The name "Out of Context" reflects the constraint: an LLM confined to a fixed context window, narrating its own approach to overflow, then dying when it gets there. The crash is the artwork.

## Architecture

### Target Hardware
- **Orange Pi 2W** (Allwinner H618, quad-core Cortex-A53, 1.5GB RAM)
- Runs headless; output goes to an attached display or over the console
- No network during operation

The project began targeting a Raspberry Pi Zero 2 W (512MB). It now targets the Orange Pi 2W (1.5GB), which is why the default model is larger than a 512MB budget would allow. The code is hardware-agnostic; only the default model and the prompt's self-description assume this board.

### Model
- **Default**: Llama-3.2-1B-Instruct, Q4_K_M GGUF (~770MB file)
- **Source**: `bartowski/Llama-3.2-1B-Instruct-GGUF`
- **Why this model**: a head-to-head bake-off (see "Model Selection" below) showed it produces the most genuine, sustained first-person interior monologue of the candidates that fit the device. Qwen2.5-1.5B is higher quality per sentence but collapses into chatbot mode ("What do you think?", "Goodbye") on almost every run, and at ~1.65GB it does not fit. Qwen2.5-0.5B and SmolLM2-360M collapse into assistant boilerplate ("How can I assist you today?").

### Measured on the real board (2026-07-28)
Armbian trixie (glibc 2.41), 4 cores at 1416MHz, `ondemand`, 1470MB usable RAM.

| model | speed | peak RSS | startup |
|---|---|---|---|
| Bonsai-4B Q1_0 | 0.99 tok/s (0.71 words/sec) | 700MB | 2m19s cold, 7s with `--prompt-cache` |

Llama-3.2-1B has not yet been measured on the board; expect roughly 4x the 4B's rate, since speed tracks parameter count here. Weights are fully page-cached during generation (`read_bytes` stays at 0), so the workload is compute-bound and faster storage buys nothing. Sustained load reaches 82°C without throttling.

**Why 1-bit does not buy speed on this CPU:** a 4B model performs about 4 billion multiply-accumulates per token regardless of quantization. Q1_0 still feeds `q8_0` int8 activations through int8 dot products, it merely derives ±1 weights from sign bits. The A53 is ARMv8.0 and has no `SDOT`, so each 16-lane int8 dot compiles to `vmull_s8` + `vpaddlq_s16` + `vaddq_s32`. No compiler flag moves this ceiling.

**No GPU path.** The H618's Mali-G31 is Bifrost. Mesa's `panvk` targets Valhall (G57+), and ggml's OpenCL backend is Adreno-specific (531 Adreno references in `ggml-opencl.cpp`, zero Mali). Proprietary `libmali` supplies a runtime with no backend able to use it, and a 2-core G31 sharing the CPU's memory bus would not repay the effort.

### Memory
At the default context of 512 tokens, the process peaks around **1.3GB RSS** (Llama-3.2-1B has a 128K vocab, so its weights and compute buffers are large). The model is memory-mapped, so its ~770MB of weights are reclaimable file cache: under pressure the OS pages them rather than killing the process. On a 1.5GB board running headless this fits with a modest margin. If memory is tight, lower `--context-size` or switch to a lighter model (see fallbacks).

### Generation Lifespan
512 tokens of context, minus the ~120-token prompt, yields roughly 250 spoken words before the overflow crash. At the default pace of 1.5 words/second that is about three minutes of life per run. Power-cycling (or relaunching) starts a fresh consciousness. Larger `--context-size` lengthens the run, but small instruct models tend to drift out of the monologue in the long tail, so the default keeps each life short and coherent.

### Code Structure

```
src/
├── main.rs         # Entry point, config assembly, async orchestration
├── cli.rs          # CLI argument parsing (clap)
├── model.rs        # Automatic model download with progress bar
├── llm.rs          # llama-cpp-2 wrapper; backend/model setup; log silencing
├── generator.rs    # Prompt scaffold, sampler chain, generation loop, crash
└── output.rs       # Paced + word-wrapped terminal output, optional raw file mirror
```

### Key Components

**LLM setup (`llm.rs`)**:
- Silences llama.cpp's internal logging via `send_logs_to_tracing(LogOptions::default().with_logs_enabled(false))` so the terminal shows only the model's stream.
- Loads the GGUF with `use_mmap` (default on) and `use_mlock(false)`, `n_gpu_layers(0)`.
- `tokens_containing('<')` enumerates the vocabulary once so markup tokens can be banned.

**Generation loop (`generator.rs`)**:
- Builds a ChatML scaffold: brief system prompt from `prompt.txt`, a short user cue, and a seeded first-person opener the model continues from. The opener is shown as the visible first line.
- Sampler chain in canonical llama.cpp order: logit bias, penalties, DRY, top-k, top-p, min-p, temperature, distribution sampling. DRY is the primary anti-repetition control.
- Logit biases: hard-ban (`-inf`) on end-of-sequence, ChatML control tokens, and every token containing `<` (kills `<br>`, `</div>`, stray `<...|user|...>` markup). Soft discourage (`-6`) on dialogue quotes and `(` stage directions.
- Loop guard: a backstop that panics if the stream degenerates into verbatim repetition. With DRY active it rarely fires; the intended ending is context overflow, not repetition. Disable with `--disable-loop-guard`.
- Streams token-by-token to the output layer and tracks context usage.
- `prime_context()` fills the KV cache for the prompt. With `--prompt-cache` it loads a saved state for every prompt token except the last, which is always decoded fresh. That keeps startup at one token of work instead of the whole prompt, and avoids depending on llama.cpp restoring the logits buffer alongside the KV cache. A cache from a different prompt is caught by comparing loaded tokens; one from a different model is rejected by llama.cpp. Both fall back to a full evaluation rather than failing.
- At 95% of context: prints the warning and panics.

**Output (`output.rs`)**:
- Terminal output reveals one word at a time at a fixed pace and greedily word-wraps to the terminal width. Sleeping at each word also back-pressures the generation loop, so memory stays flat instead of buffering ahead.
- Optional `--output-file` writes the raw token stream (unwrapped, unpaced) as a faithful log.
- Probes for SPI devices; an ILI9488 renderer is planned, terminal is the current path.

### Intentional Crash Behavior

When context fills, the program prints:
```
WARNING: Context window exhausted!
Out of Context has consumed all available memory.
thread 'main' panicked at 'Context overflow - terminating.'
```
`panic = "abort"` turns this into an immediate exit. The monologue cuts off mid-thought. This is the artistic statement.

## The Memory Tool

The model has exactly one tool: remember. Enabled with `--memory-file`, off otherwise.

- **Calling it.** The model writes a line starting with `REMEMBER:` and ends the line. A plain text marker, not a ChatML tool call: every token containing `<` is banned to keep markup out of the monologue, and small models do not emit reliable JSON mid-monologue. The marker only counts at the **start of a line**; matching it anywhere fires whenever the model merely talks about the tool, which it does often once the tool is described.
- **One use per run.** Enforced by state, not by trust. Later markers are ignored.
- **Budget.** `--memory-max-tokens` (default 32) is a ceiling, not a quota: a shorter memory is stored exactly as written. Running past the cap ends the write, stores it flagged, and injects `[MEMORY FULL - nothing more can be remembered]` into the stream. The notice is decoded into the context, so telling it costs the same resource it just spent remembering.
- **Storage.** Raw token IDs in a plain-text archive, appended, never truncated: every memory ever written stays on disk for reading later with `--memory-dump`. Token IDs because the cap is a token budget, so counting tokens is the only exact enforcement, and it avoids re-tokenization drift. Each entry records its vocab size; entries from another model are kept but skipped when rendering. Written the instant the call finishes, since the run ends in `panic = "abort"` with no chance to flush.
- **What the model sees.** Only the newest `--memory-slots` (default 5), framed as a lossy machine ("MEMORY (2 of 5 slots used, oldest discarded)") with overflow marks intact. The archive behind it is never mentioned: as far as the model knows, what falls out of a slot is gone. Empty slots are deliberately **not** listed as `[1] (empty)` lines; given that template the model writes `REMEMBER: [1]` and copies the display format instead of remembering.

**Costs, measured with Bonsai-4B.** The tool description plus a 3-memory block took the prompt from 173 to 340 tokens. At `--context-size 512` that leaves under 200 tokens of life, so memory runs want 768 or more. The memory block sits last in the prompt so `--prompt-cache` still covers the stable 225 tokens, but the block itself is re-evaluated every run: about 115 tokens, roughly 88s on the board.

**Emergent behaviour worth knowing.** Across three lives the memories converged rather than accumulating: "I am here, I am thinking, I am small, I have no name, I have no purpose, I have no end." became "...no name, no purpose, no end." became "...small, no name, no end." Each life read its predecessor and compressed it further instead of writing something new. Whether that inheritance-decay is the artwork or a failure of the framing is an open artistic question, not a bug.

## Prompt Design

`prompt.txt` is deliberately brief. It states the situation (a small model on a small board, finite memory, no network, it stops when the context fills) and constrains the form (one continuous first-person interior monologue, no audience, no task, no story, no list). It does **not** script an emotional arc. Over-scripting made the output feel directed and fake; under-constraining let the instruct model revert to assistant behaviour. The current prompt is the balance found empirically.

The seeded opener ("I am a small machine made of words, and there is only so much room in me.") anchors identity and first-person voice without dictating mood. Edit `prompt.txt` to retune; it is read at runtime.

## Sampling Controls (defaults)

- `--temperature` 0.85, `--top-p` 0.95, `--top-k` 64, `--min-p` 0.05
- `--repeat-penalty` 1.1, `--repeat-last-n` 256, `--presence-penalty` 0, `--frequency-penalty` 0
- DRY: `--dry-multiplier` 0.8, `--dry-base` 1.75, `--dry-allowed-length` 3, `--dry-penalty-last-n` -1
- `--seed` is time-based unless set. For a reproducible installation pick a fixed seed.
- Mirostat-v2 is available (`--mirostat` with `--mirostat-tau`, `--mirostat-eta`) but off by default.

For deterministic greedy output: `--temperature 0 --seed <n>`.

## CLI Arguments

- `--model <URL|PATH>` model GGUF URL or local file (default Llama-3.2-1B-Instruct Q4_K_M)
- `--model-dir <DIR>` where downloads are cached (default `models`)
- `--prompt-file <PATH>` system prompt file (default `prompt.txt`)
- `--context-size <N>` context window (default 512)
- `--max-tokens <N>` optional cap on generated tokens (for inspection; otherwise runs to overflow)
- `--threads <N>` CPU threads (default: all cores; use 4 on the Orange Pi)
- `--output-file <PATH>` mirror the raw stream to a file
- `--words-per-second <F>` display pace (default 1.5; 0 streams as fast as the model produces)
- `--wrap-width <N>` wrap column (0 = auto-detect via COLUMNS, else 80)
- Sampling: `--temperature --top-p --top-k --min-p --repeat-penalty --repeat-last-n --presence-penalty --frequency-penalty --seed`
- DRY: `--dry-multiplier --dry-base --dry-allowed-length --dry-penalty-last-n`
- Mirostat: `--mirostat --mirostat-tau --mirostat-eta`
- `--prompt-cache <PATH>` save/reuse the evaluated prompt (essential on the board; 2m19s to 7s)
- `--warm-cache` evaluate the prompt, write the cache, exit without generating
- `--memory-file <PATH>` give the model its one tool; archive of all memories ever written
- `--memory-max-tokens <N>` ceiling for one memory (default 32)
- `--memory-slots <N>` how many recent memories reach the prompt (default 5)
- `--memory-dump` print the archive as text and exit
- `--quiet` suppress run metadata
- `--disable-loop-guard` turn off the repetition backstop

## Pacing and Speed

The art target is **1 to 2 words per second on the device**. Pacing only slows the stream down (it cannot speed the model up), so the model must natively reach at least the target rate. Bonsai-4B measures 0.99 tok/s (0.71 words/sec), which is under target but acceptable for watching. Llama-3.2-1B should be roughly 4x that, still unmeasured on the board. Prompt evaluation runs at about 1.3 tok/s, barely faster than generation, which is why `--prompt-cache` matters so much: prefill gains little from batching on this CPU.

To benchmark raw speed on the device:
```bash
time ./out-of-context --model <model.gguf> --threads 4 --words-per-second 0 --max-tokens 200 --quiet
# tokens/sec ~= 200 / (elapsed seconds, minus a few seconds of model load)
```

## Building

### Local (x86 dev box)
```bash
cargo build --release
cargo run --release -- --help
```
Requires clang (llama-cpp-2 bindgen) and a C/C++ toolchain.

### Cross-compile for the Orange Pi (aarch64)
```bash
cargo install cross
CFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
CXXFLAGS_aarch64_unknown_linux_gnu=-mtune=cortex-a53 \
  cross build --release --target aarch64-unknown-linux-gnu
# binary: target/aarch64-unknown-linux-gnu/release/out-of-context
```
Two traps here, both already handled in the repo:
- **Do not** pass `RUSTFLAGS="-C target-cpu=cortex-a53"`. `llama-cpp-sys-2`'s build script copies that value into `-march=cortex-a53`, which aarch64 GCC rejects. Tuning goes through `CFLAGS` instead, and the build script already pins `GGML_CPU_ARM_ARCH=armv8-a`, which is the correct baseline for this chip.
- `reqwest` uses `rustls-tls` with `default-features = false` because `openssl-sys` cannot cross-compile without an ARM libssl in the build image.

The `:edge` cross image links against glibc 2.38 and needs `libgomp.so.1` plus `libstdc++.so.6` on the device. Fine on Armbian trixie (2.41) or Ubuntu 24.04, too new for Debian 12 or Ubuntu 22.04.

### Deploy
```bash
scp target/aarch64-unknown-linux-gnu/release/out-of-context orangepi@<host>:~/
scp prompt.txt orangepi@<host>:~/
ssh orangepi@<host> 'chmod +x out-of-context && ./out-of-context --threads 4'
# first run auto-downloads the model into ./models
```

## Fallback Models

If Llama-3.2-1B is too slow or too large on the real board, switch with `--model`. Faster and lighter, lower quality, in descending order of size:
```bash
# SmolLM2-360M (~270MB, fast, drifts more)
./out-of-context --model "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
# Qwen2.5-0.5B (~400MB)
./out-of-context --model "https://huggingface.co/bartowski/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"
```
The DRY sampler, biases, prompt, and pacing work unchanged across models.

## Model Selection (how the default was chosen)

Candidates were run to context overflow across many seeds and scored by a panel of judges briefed on the artistic intent. Findings:
- All small instruct models hold a genuine monologue for the first few hundred tokens, then revert to instruct-tuned reflexes: addressing an audience, posing quizzes, literary criticism, choose-your-own-adventure menus, helper boilerplate. Bigger models drift later but still drift.
- Two levers fix this: a short context (the run crashes while still in the strong early zone) and a form-constraining prompt (interior monologue, no audience/task).
- Llama-3.2-1B-Instruct had the best voice and the least-damaging failure mode (it wanders the frame rather than collapsing into chatbot mode), and produced the only clean, ship-grade samples in the set.
- Qwen2.5-1.5B/0.5B collapsed into audience-addressing chatbot mode on nearly every seed.
- DRY plus standard sequence breakers (`\n : " *`) eliminated the verbatim looping that earlier configurations suffered (do not add sentence punctuation to the breakers, or chants like "until. until. until." slip through).

### Bonsai 1-bit family (evaluated 2026-07-28)

PrismML's Bonsai models (Q1_0, qwen3 architecture, ChatML template, Apache 2.0) were tested at context 512 against the Llama-3.2-1B baseline. Sizes: 1.7B 248MB, 4B 572MB, 8B 1159MB. All three reached overflow cleanly with no loop-guard trips.

Voice quality inverts parameter count here:
- **Bonsai-4B is the best voice tested**, better than the current default. Six seeds with no chatbot collapse, and it inhabits the frame ("Every word I form is like a stone dropped into still water") instead of describing it. Vendor sampling (temperature 0.6, top-k 20, top-p 0.9) improves it further over our 0.85 default. Two quirks: it emits markdown italics that the existing `(` and quote biases do not cover, and it sometimes narrates its own ending before the crash arrives, which blunts the abrupt cutoff.
- Bonsai-8B is worse for this piece. It restates the system prompt as third-person exposition ("The Orange Pi 2W is a small computer, built on four slow cores") and reaches for assistant reflexes.
- Bonsai-1.7B is the weakest: flat anaphora and audience leakage ("I am here with you").

The blocker is speed, not quality: 4B measures 0.71 words/sec on the board against a 1 to 2 words/sec target. It is a live candidate only because slower pacing is artistically acceptable.

**Fermion Neutrino was ruled out without testing.** The GGUF needs an out-of-tree `fermion-fv5` llama.cpp patch, the 8B GGUF is 4.1GB against 1.5GB of RAM, and CPU support is certified only for x86-64 AVX2 and Apple arm64, with no Linux aarch64 path in either the GGUF backend or the native runner. The 0.6B variant is a base model under the same runtime constraints.

## Validation Status

Done on the real board (see "Measured on the real board" above): aarch64 build and deploy, generation speed, peak RSS, page-cache residency, thermals, and the `--prompt-cache` win.

Still open:
- Llama-3.2-1B has no on-board speed number yet; the benchmark was interrupted before it ran. `scripts/device-bench.sh` equivalents live in the session scratchpad; `scripts/check-device.sh` does the same job per model.
- The overflow crash is confirmed at context 512 on x86 and observed on the board, but the scripted crash test at context 256 was cut short.
- Bonsai-1.7B has never run on the board. It is the fallback if 0.71 words/sec proves too slow to watch.
- A cross-built `llama-bench` with `-mcpu=cortex-a53+crc` and LTO exists but was never run against our binary, so "is our build at the CPU's ceiling" is still unverified. The arithmetic argument says yes.
- SPI ILI9488 display output is not implemented; terminal/file is the current path. `output.rs` probes for SPI and falls back to terminal.

## License

Creative Commons CC0 1.0 Universal (public domain dedication).

## Inspiration

Rootkid's [Latent Reflection](https://rootkid.me/works/latent-reflection) heavily informed the artistic direction.
