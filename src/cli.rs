use clap::Parser;
use std::path::PathBuf;

/// Out of Context - An LLM text generator that runs until context exhaustion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Hugging Face model URL or path to local GGUF model file.
    ///
    /// Examples:
    ///   - "https://huggingface.co/mav23/SmolLM-360M-Instruct-GGUF/resolve/main/smollm-360m-instruct.Q3_K_M.gguf"
    ///   - "./my-model.gguf"
    #[arg(
        short,
        long,
        default_value = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
    )]
    pub model: String,

    /// Directory to store downloaded models
    #[arg(short = 'd', long, default_value = "models")]
    pub model_dir: PathBuf,

    /// Path to the system prompt file
    #[arg(short, long, default_value = "prompt.txt")]
    pub prompt_file: PathBuf,

    /// Context window size in tokens
    #[arg(short, long, default_value_t = 1024)]
    pub context_size: usize,

    /// Optional cap on generated tokens (helpful for readability)
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Number of CPU threads to use (defaults to available cores)
    #[arg(long)]
    pub threads: Option<usize>,

    /// Optional path to mirror output into a file (in addition to terminal)
    #[arg(long)]
    pub output_file: Option<PathBuf>,

    /// Sampling temperature (higher = more random, 0 = greedy)
    #[arg(long, default_value_t = 0.22)]
    pub temperature: f32,

    /// Nucleus sampling probability mass (1.0 disables filtering)
    #[arg(long, default_value_t = 0.5)]
    pub top_p: f32,

    /// Top-k sampling cap (0 disables filtering)
    #[arg(long, default_value_t = 20)]
    pub top_k: usize,

    /// Penalize recent repeats (1.0 disables)
    #[arg(long, default_value_t = 2.15)]
    pub repeat_penalty: f32,

    /// How many recent tokens to consider for repetition penalties
    #[arg(long, default_value_t = -1)]
    pub repeat_last_n: i32,

    /// Presence penalty (encourages introducing new tokens)
    #[arg(long, default_value_t = 1.35)]
    pub presence_penalty: f32,

    /// Frequency penalty (discourages repeating frequently used tokens)
    #[arg(long, default_value_t = 1.05)]
    pub frequency_penalty: f32,

    /// Random seed for sampling (omit to use a time-based seed)
    #[arg(long)]
    pub seed: Option<u32>,

    /// Override the user prompt that follows the system prompt (advanced)
    #[arg(long)]
    pub user_prompt: Option<String>,

    /// Silence run metadata and only stream the model output
    #[arg(long)]
    pub quiet: bool,

    /// Interval between anchor sentences that disrupt looping (0 to disable)
    #[arg(long, default_value_t = 80)]
    pub anchor_interval: usize,

    /// Disable anchor injection entirely
    #[arg(long)]
    pub disable_anchors: bool,

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
