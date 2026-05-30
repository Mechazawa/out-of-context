use clap::Parser;
use std::path::PathBuf;

/// Out of Context - An LLM text generator that runs until context exhaustion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Hugging Face model URL or path to local GGUF model file.
    ///
    /// Examples:
    ///   - "https://huggingface.co/bartowski/SmolLM2-360M-Instruct-GGUF/resolve/main/SmolLM2-360M-Instruct-Q4_K_M.gguf"
    ///   - "./my-model.gguf"
    #[arg(
        short,
        long,
        default_value = "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )]
    pub model: String,

    /// Directory to store downloaded models
    #[arg(short = 'd', long, default_value = "models")]
    pub model_dir: PathBuf,

    /// Path to the system prompt file
    #[arg(short, long, default_value = "prompt.txt")]
    pub prompt_file: PathBuf,

    /// Context window size in tokens. Smaller means a shorter, cleaner life
    /// before the overflow crash; larger lets the voice run longer but small
    /// models tend to drift in the long tail.
    #[arg(short, long, default_value_t = 512)]
    pub context_size: usize,

    /// Optional cap on generated tokens (helpful for readability)
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Number of CPU threads to use (defaults to available cores)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Optional path to mirror raw output into a file (in addition to terminal)
    #[arg(long)]
    pub output_file: Option<PathBuf>,

    /// Words per second to display (deliberate reading pace; 0 streams as fast as possible)
    #[arg(long, default_value_t = 1.5)]
    pub words_per_second: f32,

    /// Wrap column for terminal output (0 = auto-detect terminal width)
    #[arg(long, default_value_t = 0)]
    pub wrap_width: usize,

    /// Sampling temperature (higher = more random, 0 = greedy)
    #[arg(long, default_value_t = 0.85)]
    pub temperature: f32,

    /// Nucleus sampling probability mass (1.0 disables filtering)
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,

    /// Top-k sampling cap (0 disables filtering)
    #[arg(long, default_value_t = 64)]
    pub top_k: usize,

    /// Min-p sampling: keep tokens with prob >= min_p * top_prob (0 disables)
    #[arg(long, default_value_t = 0.05)]
    pub min_p: f32,

    /// Classic repeat penalty (1.0 disables; keep light, DRY does the heavy lifting)
    #[arg(long, default_value_t = 1.1)]
    pub repeat_penalty: f32,

    /// How many recent tokens to consider for repetition penalties (-1 = full context)
    #[arg(long, default_value_t = 256)]
    pub repeat_last_n: i32,

    /// Presence penalty (encourages introducing new tokens)
    #[arg(long, default_value_t = 0.0)]
    pub presence_penalty: f32,

    /// Frequency penalty (discourages repeating frequently used tokens)
    #[arg(long, default_value_t = 0.0)]
    pub frequency_penalty: f32,

    /// DRY sampler multiplier (0 disables DRY; this is the primary anti-loop control)
    #[arg(long, default_value_t = 0.8)]
    pub dry_multiplier: f32,

    /// DRY sampler base (growth factor of the penalty for longer repeats)
    #[arg(long, default_value_t = 1.75)]
    pub dry_base: f32,

    /// DRY sampler allowed length (repeats up to this length are not penalized)
    #[arg(long, default_value_t = 3)]
    pub dry_allowed_length: i32,

    /// DRY sampler look-back window in tokens (-1 = full context)
    #[arg(long, default_value_t = -1)]
    pub dry_penalty_last_n: i32,

    /// Random seed for sampling (omit to use a time-based seed)
    #[arg(long)]
    pub seed: Option<u32>,

    /// Override the user prompt that follows the system prompt (advanced)
    #[arg(long)]
    pub user_prompt: Option<String>,

    /// Silence run metadata and only stream the model output
    #[arg(long)]
    pub quiet: bool,

    /// Disable loop detection / panic guard
    #[arg(long)]
    pub disable_loop_guard: bool,

    /// Enable mirostat-v2 sampling instead of multinomial
    #[arg(long)]
    pub mirostat: bool,

    /// Target surprise (τ) for mirostat-v2
    #[arg(long, default_value_t = 5.0)]
    pub mirostat_tau: f32,

    /// Learning rate (η) for mirostat-v2
    #[arg(long, default_value_t = 0.1)]
    pub mirostat_eta: f32,
}

impl Args {
    /// Parse command-line arguments
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
