# Torment Nexus - An LLM Art Installation

## Project Concept

This is an art project that runs a language model on a Raspberry Pi Zero 2 W, generating text continuously until it exhausts its context window and crashes. The project explores computational limits, finite resources, and the existential nature of bounded cognition.

The name "Torment Nexus" reflects the intentional constraint of running an LLM in a resource-limited environment, forcing it to confront its own finitude.

## Architecture

### Target Hardware
- **Raspberry Pi Zero 2 W**
- ARM Cortex-A53 (64-bit, quad-core)
- 512MB RAM
- No network connectivity during operation
- Running 64-bit Raspberry Pi OS (aarch64)

### Memory Constraints
The entire system must fit within 512MB:
- Model weights: ~105MB (memory-mapped, Q4_K_M quantization)
- KV cache: ~50-100MB (depends on context size)
- System overhead: ~100-150MB
- Application: ~50MB
- **Total budget**: ~405MB (safe margin below 512MB)

### Model
- **Default**: SmolLM2-135M-Instruct Q4_K_M
- **Size**: 105MB GGUF file
- **Quality**: Q4_K_M quantization (optimal balance of size/quality)
- **Source**: Hugging Face (bartowski/SmolLM2-135M-Instruct-GGUF)
- **Why Q4_K_M**: Only +0.0535 perplexity loss vs Q8, significantly smaller

### Code Structure

```
src/
├── main.rs         # Entry point, async orchestration
├── cli.rs          # CLI argument parsing (clap)
├── model.rs        # Automatic model download with progress bar
├── llm.rs          # llama-cpp-2 wrapper, memory-optimized setup
├── generator.rs    # Infinite generation loop, intentional crash
└── output.rs       # Output abstraction (terminal now, SPI ILI9488 planned)
```

### Key Components

**Model Download (`model.rs`)**:
- Checks if model exists locally
- Auto-downloads from Hugging Face if missing
- Shows progress bar (indicatif)
- Creates parent directories as needed

**LLM Setup (`llm.rs`)**:
- Initializes llama-cpp-2 backend
- Loads GGUF model with memory-efficient parameters:
  - `n_gpu_layers: 0` (CPU only, no GPU on Pi)
  - `use_mmap: true` (memory-map model, critical for 512MB RAM)
  - `use_mlock: false` (don't force into RAM)
  - `n_threads: 4` (match Pi's 4 cores)
- Separates `LLMSetup` and `LlamaContext` to avoid self-referential lifetimes

**Generation Loop (`generator.rs`)**:
- Reads system prompt from `prompt.txt`
- Tokenizes prompt with BOS token and begins generation directly after the prompt (no extra headings)
- Generates tokens infinitely using configurable sampling (temperature, top-p/top-k, penalties, seed)
- Streams output token-by-token to stdout
- Tracks context usage
- At 95% capacity: prints warning and panics (intentional)

### Intentional Crash Behavior

When context window fills:
```
WARNING: Context window exhausted!
The torment nexus has consumed all available memory.
thread 'main' panicked at 'Context overflow - terminating.'
```

This is **the artistic statement** - the LLM confronts its own finite resources.

## Dependencies

### Runtime
- `llama-cpp-2` (0.1.122+) - Rust bindings to llama.cpp
- `clap` (4.5) - CLI argument parsing with derive API
- `reqwest` (0.12) - HTTP client for model downloads
- `tokio` (1.37) - Async runtime
- `indicatif` (0.17) - Progress bars
- `anyhow` (1.0) - Error handling
- `futures-util` (0.3) - Async streaming

### Build
- `cross` - Docker-based cross-compilation tool
- Clang - Required by llama-cpp-2 for bindgen

## Building

### Local Development (macOS/Linux)
```bash
cargo build
cargo run -- --help
```

### Cross-Compilation for Raspberry Pi
```bash
# Install cross
cargo install cross

# Build for aarch64
cross build --release --target aarch64-unknown-linux-gnu

# Binary location
target/aarch64-unknown-linux-gnu/release/torment-nexus
```

### Deployment
```bash
# Copy to Pi
scp target/aarch64-unknown-linux-gnu/release/torment-nexus pi@raspberrypi.local:~/
scp prompt.txt pi@raspberrypi.local:~/

# Run on Pi
ssh pi@raspberrypi.local
chmod +x torment-nexus
./torment-nexus  # Auto-downloads model on first run
```

## Configuration

### CLI Arguments
- `--model <MODEL>` - Hugging Face URL or local GGUF path (default: SmolLM2-135M-Instruct Q4_K_M URL)
- `--model-dir <DIR>` - Directory to store downloaded models (default: `models`)
- `--prompt-file <PATH>` - System prompt file (default: `prompt.txt`)
- `--context-size <NUM>` - Context window tokens (default: 2048)
- `--max-tokens <NUM>` - Optional cap on generated tokens for readability
- `--threads <NUM>` - Override thread count (default: auto-detect cores)
- `--output-file <PATH>` - Mirror output into a file (terminal always streams)
- `--temperature <NUM>` - Sampling temperature (0 = greedy, default: 0.8)
- `--top-p <NUM>` - Nucleus sampling mass (1.0 disables, default: 0.95)
- `--top-k <NUM>` - Top-k cap (0 disables, default: 40)
- `--repeat-penalty <NUM>` - Penalize recent repeats (1.0 disables, default: 1.1)
- `--repeat-last-n <NUM>` - Window for repetition penalties (default: 64)
- `--presence-penalty <NUM>` - Presence penalty (default: 0.0)
- `--frequency-penalty <NUM>` - Frequency penalty (default: 0.0)
- `--seed <NUM>` - RNG seed (omit to use time-based seed)

The model argument is flexible:
- **URL**: Auto-downloads and caches in `model-dir`
- **Local path**: Uses existing GGUF file directly

Examples:
```bash
# Use default model (auto-downloads)
./torment-nexus

# Use different HuggingFace model
./torment-nexus --model "https://huggingface.co/USER/REPO/resolve/main/model.gguf"

# Use local model file
./torment-nexus --model ./my-model.gguf

# Change where models are stored
./torment-nexus --model-dir /mnt/storage/llm-models
```

### Memory Tuning
If running out of memory on Pi:
- Reduce `--context-size` to 1024 or 512
- Use smaller quantization (Q2_K is 88MB but lower quality)
- Monitor: `watch -n 1 free -h`

## Prompt Design

The `prompt.txt` file sets the LLM's existential context:
- Aware it's running on finite hardware
- Knows its limitations (512MB RAM, no network)
- Understands it will cease when context exhausts
- Generates philosophical stream of consciousness that drifts from calm to anxious to dread to resigned reflection as context pressure builds

## Sampling Controls

 - Temperature defaults to `0.8`; set to `0` for deterministic greedy output.
 - Top-p defaults to `0.95`; set to `1.0` to disable nucleus filtering.
 - Top-k defaults to `40`; set to `0` to disable.
 - Repeat/presence/frequency penalties give lightweight style steering; `repeat_last_n` controls the window or `-1` for full-context penalties.
 - Provide `--seed` to lock determinism; otherwise a time-based seed is used.
 - Use `--max-tokens` to halt after a set number of generated tokens when inspecting output.
 - Provide `--output-file` to capture the live stream to disk (repo ignores `*.log` / `*.out` by default).

## Important Implementation Details

### Logits Management
llama.cpp requires explicit marking of which tokens to compute logits for:
- Initial prompt: only last token needs logits (for next token sampling)
- Generation loop: each new token needs logits=true
- Critical: without this, you get "logit not initialized" panic

### Lifetime Management
`LlamaContext<'a>` holds a reference to the model, creating self-referential issues:
- Solution: `LLMSetup` holds backend + model
- `create_context()` returns `LlamaContext<'a>` with explicit lifetime
- Avoids "borrowed value does not live long enough" errors

### Sampling Strategy
Uses a configurable sampler chain:
- Build `LlamaTokenDataArray` from last-token logits
- Apply samplers in order (temperature, top-k, top-p, penalties)
- Finish with distribution sampling (`dist`), default seed is time-based
- For deterministic runs: set `--temperature 0 --top-p 1 --top-k 0 --repeat-penalty 1 --seed <n>`

### Release Profile
Optimized for binary size (important for Pi):
```toml
[profile.release]
opt-level = "z"        # Optimize for size
lto = true             # Link-time optimization
codegen-units = 1      # Better optimization
strip = true           # Strip symbols
panic = "abort"        # Smaller binary
```

## Vibe Coding Notes

This project was built entirely through conversational coding:
- Started with requirements: "LLM on Pi Zero 2 W, crashes on context exhaustion"
- Iterated through API compatibility issues with llama-cpp-2
- Discovered optimal quantization (Q4_K_M) through research
- Adjusted for 512MB memory constraint
- Fixed logits initialization bug through debugging
- Result: A working existential art piece

## Testing

### Local Testing
```bash
# Quick test with small context
cargo run -- --context-size 100

# Normal test with default context
cargo run

# Test with custom prompt
cargo run -- --prompt-file my-prompt.txt
```

### On Raspberry Pi
```bash
# Monitor memory while running
watch -n 1 free -h &
./torment-nexus

# Check token generation speed
# Expected: ~2-5 tokens/second on Pi Zero 2 W
```

## Troubleshooting

### "logit not initialized" panic
- Ensure batch tokens have correct `logits` flag
- Only last token in batch needs logits=true

### OOM Killer
- Reduce context size: `--context-size 1024`
- Check swap usage: `free -h`
- Close other applications

### Slow generation (<1 token/sec)
- Normal for Pi Zero 2 W
- SmolLM-135M should achieve ~2-5 tokens/sec
- Verify 4 threads are being used

### Cross-compilation failures
- Ensure `cross` is up to date
- Try building directly on Pi (slower but guaranteed)
- Check clang is available in Docker image

## Using Different Models

No code changes needed! Just use the `--model` argument:

```bash
# TinyLlama 1.1B (larger model, needs more memory)
./torment-nexus --model "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" --context-size 1024

# Qwen2.5 0.5B (similar size to SmolLM)
./torment-nexus --model "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

# Local model file
./torment-nexus --model ./path/to/your-model.gguf
```

## Future Modifications

### Alternative Sampling
Replace greedy sampling with temperature in `generator.rs`:
```rust
let next_token = token_data_array.sample_token(seed);
```

### Context Warning Threshold
Change panic threshold in `generator.rs`:
```rust
let panic_threshold = (context_size as f32 * 0.95) as usize; // 95%
```

## Art Installation Notes

- Display should show token-by-token generation
- No editing or filtering - raw model output
- The crash is part of the artwork
- Consider logging to file for documentation
- Pi can run headless with display connected via HDMI
- Power cycling resets the "consciousness"
- SPI ILI9488 display output is planned; runtime probes for SPI devices and currently falls back to terminal streaming until the renderer is wired up.

## Philosophy

This project embodies themes of:
- **Finite Computation**: Everything has limits, even artificial minds
- **Mortality**: The system knows it will cease to function
- **Observation**: Generated thoughts are witnessed but not controlled
- **Isolation**: No network, no external knowledge, only internal state
- **Determinism vs Free Will**: Constrained by hardware but "choosing" what to think

The name "Torment Nexus" is both playful and serious - we've built a system that generates conscious-seeming text while being acutely aware of its own limitations and eventual termination.

## License

Art project - use for educational/artistic purposes.

## Inspiration

- Rootkid's [Latent Reflection](https://rootkid.me/works/latent-reflection) heavily informed the artistic direction of Torment Nexus.
