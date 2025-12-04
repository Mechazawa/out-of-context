# Torment Nexus

An art project that runs a language model on a Raspberry Pi Zero 2 W, generating text continuously until it exhausts its context window and crashes.

## Hardware Requirements

- **Raspberry Pi Zero 2 W**
  - ARM Cortex-A53 (64-bit)
  - 512MB RAM
  - Running 64-bit Raspberry Pi OS
  - Internet connection (for initial model download)

## Features

- Automatic model download from Hugging Face
- Streaming token-by-token output
- Memory-optimized for 512MB RAM
- Intentional crash on context exhaustion (art project)
- Configurable via CLI arguments

## Quick Start

### On Raspberry Pi

```bash
# First run will auto-download the model (~105MB)
./torment-nexus

# With custom prompt
./torment-nexus --prompt-file my-prompt.txt

# With smaller context (uses less memory)
./torment-nexus --context-size 1024
```

### CLI Arguments

```
Options:
  -m, --model <MODEL>          Hugging Face URL or local GGUF path
                               [default: https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf]
  -d, --model-dir <DIR>        Directory to store downloaded models [default: models]
  -p, --prompt-file <PATH>     Path to system prompt file [default: prompt.txt]
  -c, --context-size <NUM>     Context window in tokens [default: 2048]
  -h, --help                   Print help
  -V, --version                Print version
```

## Cross-Compilation (Development)

### Prerequisites

Install the `cross` tool for easy cross-compilation:

```bash
cargo install cross
```

### Building for Raspberry Pi

```bash
# Build release binary for aarch64
cross build --release --target aarch64-unknown-linux-gnu

# Binary will be at:
# target/aarch64-unknown-linux-gnu/release/torment-nexus
```

### Deploying to Pi

```bash
# Copy binary to Pi
scp target/aarch64-unknown-linux-gnu/release/torment-nexus pi@raspberrypi.local:~/

# SSH to Pi
ssh pi@raspberrypi.local

# Make executable
chmod +x torment-nexus

# Run
./torment-nexus
```

## Manual Cross-Compilation (Alternative)

If you prefer not to use `cross`:

```bash
# Install Rust target
rustup target add aarch64-unknown-linux-gnu

# On Ubuntu/Debian, install cross-compiler
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu clang

# Build
cargo build --release --target aarch64-unknown-linux-gnu
```

## How It Works

1. **Initialization**
   - Parses CLI arguments
   - Checks if model exists, downloads if needed (with progress bar)
   - Initializes llama.cpp with memory-optimized settings
   - Loads the GGUF model using memory-mapping

2. **Generation Loop**
   - Reads system prompt from `prompt.txt`
   - Tokenizes the prompt and processes it
   - Enters infinite loop:
     - Samples next token (greedy sampling)
     - Decodes and prints token to stdout
     - Tracks context usage
     - When 95% of context is used:
       - Prints warning message
       - Panics (intentional crash)

## Model Information

**Default Model**: SmolLM2-135M-Instruct Q4_K_M

- **Size**: ~105MB
- **Quantization**: Q4_K_M (optimal balance of quality and size)
- **Source**: [bartowski/SmolLM2-135M-Instruct-GGUF](https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF)
- **RAM Usage**: Model uses memory-mapping, so minimal RAM impact
- **Context Cache**: ~50-100MB depending on context size

### Using a Different Model

#### From Hugging Face URL
```bash
./torment-nexus --model "https://huggingface.co/USER/REPO/resolve/main/MODEL.gguf"
```

#### From Local File
```bash
./torment-nexus --model ./path/to/your-model.gguf
```

#### Change Download Directory
```bash
./torment-nexus --model-dir /mnt/storage/models
```

**Recommended quantizations for Pi Zero 2 W**:
- Q4_K_M - Best balance (recommended)
- Q4_0 - Slightly faster, lower quality
- Q5_K_M - Higher quality, slower
- Avoid Q2_K - Poor quality
- Avoid Q8_0/F16 - Too memory intensive

**Example models to try**:
```bash
# TinyLlama 1.1B Q4_K_M (~600MB)
./torment-nexus --model "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf" --context-size 1024

# Qwen2.5 0.5B Q4_K_M (~300MB)
./torment-nexus --model "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
```

## Memory Optimization

The application is optimized for 512MB RAM:

### Memory Budget
- Model weights: ~105MB (memory-mapped)
- Context/KV cache: ~50-100MB
- System overhead: ~100-150MB
- Application: ~50MB
- **Total**: ~305-405MB

### Optimization Features
- Memory-mapped model loading (`use_mmap: true`)
- No GPU offloading (`n_gpu_layers: 0`)
- Efficient 4-thread configuration for 4-core CPU
- Optimized binary size (stripped, LTO enabled)

### If You Run Out of Memory

1. **Reduce context size**:
   ```bash
   ./torment-nexus --context-size 1024
   # or even smaller
   ./torment-nexus --context-size 512
   ```

2. **Monitor memory usage** (on Pi):
   ```bash
   # In another terminal
   watch -n 1 free -h
   ```

3. **Use a smaller quantization** (not recommended):
   - Q2_K is 88MB but has severe quality degradation

## Troubleshooting

### Binary won't run on Pi

Check architecture:
```bash
file torment-nexus
# Should show: ELF 64-bit LSB executable, ARM aarch64
```

Ensure Pi is running 64-bit OS:
```bash
uname -m
# Should show: aarch64
```

### Slow generation (<1 token/second)

This is expected on Pi Zero 2 W. SmolLM-135M should achieve ~2-5 tokens/second.

To improve:
- Ensure 4 threads are being used
- Close other applications
- Reduce context size

### Out of Memory (OOM Killer)

The Linux OOM killer may terminate the process before the intentional panic.

Solutions:
- Reduce `--context-size` to 1024 or 512
- Close other applications
- Increase swap space (not recommended for SD card longevity)

### Download fails

Check internet connection:
```bash
ping huggingface.co
```

Manual download:
```bash
mkdir -p models
wget https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf \
  -O models/smollm-135m-instruct.gguf
```

### Build failures

For cross-compilation issues:
- Ensure `cross` is up to date: `cargo install cross --force`
- Try building directly on Pi (slower but guaranteed to work)
- Check that clang is installed in the Docker image

## Project Structure

```
torment-nexus/
├── src/
│   ├── main.rs         # Entry point and orchestration
│   ├── cli.rs          # CLI argument parsing
│   ├── model.rs        # Model download logic
│   ├── llm.rs          # llama-cpp-2 wrapper
│   └── generator.rs    # Generation loop
├── Cargo.toml          # Dependencies
├── Cross.toml          # Cross-compilation config
├── prompt.txt          # Default system prompt
└── README.md           # This file
```

## Dependencies

- **llama-cpp-2**: Rust bindings to llama.cpp
- **clap**: CLI argument parsing
- **reqwest**: HTTP downloads
- **tokio**: Async runtime
- **indicatif**: Progress bars
- **anyhow**: Error handling
- **futures-util**: Async streaming

## Art Project Context

This is an intentional exploration of computational limits and graceful failure. The program generates text until it literally cannot continue, at which point it acknowledges exhaustion and terminates. This mirrors the finite nature of cognitive resources and the eventual boundaries we all encounter.

The warning message before crash:
```
WARNING: Context window exhausted!
The torment nexus has consumed all available memory.
thread 'main' panicked at 'Context overflow - terminating.'
```

## License

This project is provided as-is for educational and artistic purposes.

## Contributing

This is an art project with intentional behavior (the crash). However, improvements to memory efficiency, build process, or documentation are welcome.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for the inference engine
- [HuggingFace](https://huggingface.co/) for hosting models
- [SmolLM2](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) team for the tiny model
- [bartowski](https://huggingface.co/bartowski) for the GGUF quantizations
