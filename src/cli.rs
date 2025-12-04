use clap::Parser;
use std::path::PathBuf;

/// Torment Nexus - An LLM text generator that runs until context exhaustion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Path to the GGUF model file
    #[arg(short, long, default_value = "models/smollm-135m-instruct.gguf")]
    pub model_path: PathBuf,

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
