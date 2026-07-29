use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::{LogOptions, send_logs_to_tracing};
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::token::LlamaToken;
use std::num::NonZeroU32;
use std::path::Path;

/// The GPU backend this build can offload to, if any. Metal needs no cargo
/// feature: llama.cpp defaults `GGML_METAL` on for every Apple target, so a
/// macOS build always links it. Everywhere else the backend has to be asked for.
const GPU_BACKEND: Option<&str> = if cfg!(target_os = "macos") {
    Some("Metal")
} else if cfg!(feature = "vulkan") {
    Some("Vulkan")
} else {
    None
};

/// Wrapper around the LLM components
/// The backend and model are stored together, and the context is created separately
/// to avoid self-referential struct issues
pub struct LLMSetup {
    pub backend: LlamaBackend,
    pub model: LlamaModel,
}

impl LLMSetup {
    /// Initialize the LLM backend and load the model.
    ///
    /// `gpu_layers` is a development convenience for iterating on prompts and
    /// framings quickly. The target board has no usable GPU, so a deployed run
    /// always passes 0, and offloading does nothing unless the build has a
    /// backend for it (see `GPU_BACKEND`).
    pub fn with_gpu_layers(model_path: &Path, gpu_layers: u32) -> Result<Self> {
        // Silence llama.cpp's verbose internal logging so the only thing on the
        // terminal is the model's stream of consciousness. (Routes logs to
        // tracing with logging disabled, i.e. dropped.)
        send_logs_to_tracing(LogOptions::default().with_logs_enabled(false));

        println!("Initializing llama.cpp backend...");

        // Those same silenced logs are where llama.cpp would say whether it
        // accepted the offload, so say it here instead.
        if gpu_layers > 0 {
            match GPU_BACKEND {
                Some(backend) => println!("Offloading up to {gpu_layers} layers to {backend}."),
                None => eprintln!(
                    "--gpu-layers {gpu_layers} does nothing: this build has no GPU backend \
                     (rebuild with --features vulkan)."
                ),
            }
        }

        // Initialize backend (this must be done first)
        let backend = LlamaBackend::init().context("Failed to initialize llama.cpp backend")?;

        // Configure model parameters for memory efficiency
        // Note: mmap is enabled by default in llama.cpp
        let model_params = LlamaModelParams::default()
            .with_n_gpu_layers(gpu_layers)
            .with_use_mlock(false); // Don't lock model in RAM

        println!("Loading model from: {}", model_path.display());

        // Load the GGUF model
        let model = LlamaModel::load_from_file(&backend, model_path, &model_params)
            .context("Failed to load model")?;

        println!("Model loaded successfully!");

        Ok(Self { backend, model })
    }

    /// Create a context for this model
    pub fn create_context<'a>(
        &'a self,
        context_size: usize,
        n_threads: usize,
    ) -> Result<LlamaContext<'a>> {
        // Configure context parameters
        let n_ctx =
            NonZeroU32::new(context_size as u32).context("Context size must be non-zero")?;

        let n_threads: i32 = n_threads
            .try_into()
            .context("Thread count is too large for llama.cpp")?;

        let context_params = LlamaContextParams::default()
            .with_n_ctx(Some(n_ctx)) // Context window size
            .with_n_threads(n_threads) // Allow tuning thread count
            .with_n_threads_batch(n_threads); // Batch processing threads

        println!(
            "Creating context with {} tokens ({} threads)...",
            context_size, n_threads
        );

        // Create context
        let context = self
            .model
            .new_context(&self.backend, context_params)
            .context("Failed to create context")?;

        println!("LLM initialization complete!");

        Ok(context)
    }

    /// Tokenize text into tokens
    pub fn tokenize(&self, text: &str, add_bos: bool) -> Result<Vec<LlamaToken>> {
        let add_bos = if add_bos {
            AddBos::Always
        } else {
            AddBos::Never
        };
        self.model
            .str_to_token(text, add_bos)
            .context("Failed to tokenize text")
    }

    /// Decode token back to text.
    ///
    /// `false` renders control tokens as nothing rather than as their literal
    /// text, which keeps the scaffold out of the stream if one is ever sampled.
    pub fn decode_token(&self, token: LlamaToken) -> Result<String> {
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        self.model
            .token_to_piece(token, &mut decoder, false, None)
            .context("Failed to decode token")
    }

    pub fn vocab_size(&self) -> i32 {
        self.model.n_vocab()
    }

    /// Every token whose plaintext contains `ch`. Used once at startup to ban
    /// markup tokens (e.g. "<br", "</i>") that web/code-trained models emit.
    pub fn tokens_containing(&self, ch: char) -> Vec<LlamaToken> {
        self.model
            // `false` = do not render control tokens; they are banned by name
            // elsewhere, so this scan only needs the plaintext vocabulary.
            .tokens(false)
            .filter_map(|(token, text)| match text {
                Ok(text) if text.contains(ch) => Some(token),
                _ => None,
            })
            .collect()
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
