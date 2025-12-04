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
└── generator.rs    # Infinite generation loop, intentional crash
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
- Tokenizes prompt with BOS token
- Generates tokens infinitely using greedy sampling
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
- `--model-path <PATH>` - Model file path (default: `models/smollm-135m-instruct.gguf`)
- `--prompt-file <PATH>` - System prompt file (default: `prompt.txt`)
- `--context-size <NUM>` - Context window tokens (default: 2048)

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
- Generates philosophical stream of consciousness

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
Uses greedy sampling (simplest, fastest):
- Get candidates from last token's logits
- Create `LlamaTokenDataArray`
- Call `sample_token_greedy()`
- For better quality: use temperature sampling (but slower)

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

## Future Modifications

### Different Models
To use different models, change the URL in `model.rs`:
```rust
const DEFAULT_MODEL_URL: &str = "https://huggingface.co/.../model.gguf";
```

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
