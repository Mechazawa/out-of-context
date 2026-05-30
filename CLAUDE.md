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
- `--quiet` suppress run metadata
- `--disable-loop-guard` turn off the repetition backstop

## Pacing and Speed

The art target is **1 to 2 words per second on the device**. Pacing only slows the stream down (it cannot speed the model up), so the model must natively reach at least the target rate. On the Orange Pi 2W, Llama-3.2-1B Q4 is expected to run roughly 2 to 3.5 tokens/second (about 1.5 to 2.5 words/second), so the default pace of 1.5 words/second yields a steady cadence with headroom. This needs confirming on the real board (see "Validation pending").

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
cross build --release --target aarch64-unknown-linux-gnu
# binary: target/aarch64-unknown-linux-gnu/release/out-of-context
```

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

## Validation Pending (next session)

- Confirm on a real Orange Pi 2W: raw tokens/second (must clear ~1.5 words/sec), peak RSS fits 1.5GB headless, and the overflow crash behaves on aarch64. Fall back to a lighter model if speed or memory fail.
- SPI ILI9488 display output is not implemented; terminal/file is the current path. `output.rs` probes for SPI and falls back to terminal.

## License

Creative Commons CC0 1.0 Universal (public domain dedication).

## Inspiration

Rootkid's [Latent Reflection](https://rootkid.me/works/latent-reflection) heavily informed the artistic direction.
