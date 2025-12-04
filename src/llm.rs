use anyhow::{Context, Result};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel, Special};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;

/// Wrapper around the LLM components
/// The backend and model are stored together, and the context is created separately
/// to avoid self-referential struct issues
pub struct LLMSetup {
    pub backend: LlamaBackend,
    pub model: LlamaModel,
}

impl LLMSetup {
    /// Initialize the LLM backend and load the model
    pub fn new(model_path: &Path) -> Result<Self> {
        println!("Initializing llama.cpp backend...");

        // Initialize backend (this must be done first)
        let backend = LlamaBackend::init()
            .context("Failed to initialize llama.cpp backend")?;

        // Configure model parameters for memory efficiency
        // Note: mmap is enabled by default in llama.cpp
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(0) // CPU only (no GPU on Pi)
            .with_use_mlock(false); // Don't lock model in RAM

        println!("Loading model from: {}", model_path.display());

        // Load the GGUF model
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("Failed to load model")?;

        println!("Model loaded successfully!");

        Ok(Self { backend, model })
    }

    /// Create a context for this model
    pub fn create_context<'a>(&'a self, context_size: usize) -> Result<LlamaContext<'a>> {
        // Configure context parameters
        let n_ctx = NonZeroU32::new(context_size as u32)
            .context("Context size must be non-zero")?;

        let context_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx)) // Context window size
            .with_n_threads(4) // Pi Zero 2 W has 4 cores
            .with_n_threads_batch(4); // Batch processing threads

        println!("Creating context with {} tokens...", context_size);

        // Create context
        let context = self.model
            .new_context(&self.backend, context_params)
            .context("Failed to create context")?;

        println!("LLM initialization complete!");

        Ok(context)
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
        let add_bos = if add_bos { AddBos::Always } else { AddBos::Never };
        self.model
            .str_to_token(text, add_bos)
            .context("Failed to tokenize text")
    }

    /// Decode token back to text
    pub fn decode_token(&self, token: LlamaToken) -> Result<String> {
        self.model
            .token_to_str(token, Special::Plaintext)
            .context("Failed to decode token")
    }
}

pub struct LlamaBatchWrapper<'a> {
    batch: LlamaBatch<'a>,
}

impl<'a> LlamaBatchWrapper<'a> {
    /// Create a new batch
    pub fn new(n_tokens: usize) -> Result<Self> {
        let batch = LlamaBatch::new(n_tokens, 1);
        Ok(Self { batch })
    }

    /// Get a mutable reference to the underlying batch
    pub fn get_mut(&mut self) -> &mut LlamaBatch<'a> {
        &mut self.batch
    }
}
