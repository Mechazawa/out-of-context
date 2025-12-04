use clap::Parser;
use std::path::PathBuf;

/// Torment Nexus - An LLM text generator that runs until context exhaustion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Hugging Face model URL or path to local GGUF model file.
    ///
    /// Examples:
    ///   - "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf"
    ///   - "./my-model.gguf"
    #[arg(short, long, default_value = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf")]
    pub model: String,

    /// Directory to store downloaded models
    #[arg(short = 'd', long, default_value = "models")]
    pub model_dir: PathBuf,

    /// Path to the system prompt file
    #[arg(short, long, default_value = "prompt.txt")]
    pub prompt_file: PathBuf,

    /// Context window size in tokens
    #[arg(short, long, default_value_t = 2048)]
    pub context_size: usize,
}

impl Args {
    /// Parse command-line arguments
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
