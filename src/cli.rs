use clap::{Parser, ValueEnum};
use std::path::PathBuf;

/// Where each life's first line comes from.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
pub enum OpenerMode {
    /// The same line every life. The repetition is the same mind booting into the
    /// same first thought, with only the memory differing.
    Fixed,
    /// One line drawn from --opener-file per life, so each life sounds like a
    /// different instance waking.
    Pool,
    /// Continue the sentence the previous life died inside. Needs --memory-file,
    /// and falls back to the first pool line on the first ever run.
    Memory,
    /// No opener. The model starts cold from the prompt, which is the most varied
    /// and the most likely to drift out of the monologue.
    None,
}

/// Out of Context - An LLM text generator that runs until context exhaustion
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Hugging Face model URL or path to local GGUF model file.
    ///
    /// Defaults to Bonsai-4B Q1_0, which has the best interior voice of anything
    /// tested and is the only model whose memory tool works. Examples:
    ///   - "https://huggingface.co/prism-ml/Bonsai-1.7B-gguf/resolve/main/Bonsai-1.7B-Q1_0.gguf"
    ///   - "./my-model.gguf"
    #[arg(
        short,
        long,
        default_value = "https://huggingface.co/prism-ml/Bonsai-4B-gguf/resolve/main/Bonsai-4B-Q1_0.gguf"
    )]
    pub model: String,

    /// Directory to store downloaded models
    #[arg(short = 'd', long, default_value = "models")]
    pub model_dir: PathBuf,

    /// Path to the system prompt file
    #[arg(short, long, default_value = "prompt.txt")]
    pub prompt_file: PathBuf,

    /// Total context window in tokens, prompt included. Ignored when
    /// --monologue-context-size is given.
    #[arg(short, long, default_value_t = 512)]
    pub context_size: usize,

    /// Tokens reserved for the monologue itself, on top of whatever the prompt
    /// and the memory block cost. Keeps every life the same length as memories
    /// accumulate, instead of the prompt eating into it.
    #[arg(long)]
    pub monologue_context_size: Option<usize>,

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

    /// Where each life's first line comes from.
    #[arg(long, value_enum, default_value_t = OpenerMode::Fixed)]
    pub opener: OpenerMode,

    /// Pool of first lines for --opener pool, one per line.
    #[arg(long, default_value = "openers.txt")]
    pub opener_file: PathBuf,

    /// Override the user prompt that follows the system prompt (advanced)
    #[arg(long)]
    pub user_prompt: Option<String>,

    /// Cache the evaluated prompt to this file so later runs skip prompt
    /// processing. On the Orange Pi the prompt costs over two minutes before
    /// the first word appears; with a cache it is one token of work. The file
    /// is written on the first run and revalidated against the model and the
    /// exact prompt tokens on every later run.
    #[arg(long)]
    pub prompt_cache: Option<PathBuf>,

    /// Evaluate the prompt, write the prompt cache, and exit without
    /// generating. Lets a supervisor pay the evaluation cost between lives so
    /// the visible run always starts immediately. Requires --prompt-cache.
    #[arg(long)]
    pub warm_cache: bool,

    /// Give the model one tool: remember. It may write a single memory per run,
    /// appended to this log as plain text, one memory per line. The newest
    /// --memory-slots memories are shown to the next run. Omit to run without
    /// memory.
    #[arg(long)]
    pub memory_file: Option<PathBuf>,

    /// Token budget for one memory. The model is told this number. Writing past
    /// it interrupts the call and the stored memory is marked as overflowed.
    #[arg(long, default_value_t = 32)]
    pub memory_max_tokens: usize,

    /// How many of the newest memories the model is shown. The file keeps every
    /// memory ever written; this only controls how many reach the prompt.
    #[arg(long, default_value_t = 5)]
    pub memory_slots: usize,

    /// How the tool and the remembered lines are described to the model. This is
    /// the artistic dial; see memory-prompt.txt. Falls back to a built-in framing
    /// when the file is absent.
    #[arg(long, default_value = "memory-prompt.txt")]
    pub memory_prompt_file: PathBuf,

    /// What the model writes to start a memory. Any token containing `<` is
    /// normally banned to keep markup out of the monologue; the exact tokens
    /// spelling this marker are exempted, so a marker like `<` works without
    /// reopening that ban.
    #[arg(long, default_value = "REMEMBER:")]
    pub memory_marker: String,

    /// What ends a memory. Empty means the write ends at the end of a sentence, at
    /// a newline, or at the token cap. Set it to close a delimiter pair, for
    /// instance `--memory-marker '<' --memory-end '>'`.
    #[arg(long, default_value = "")]
    pub memory_end: String,

    /// How much of a remembered line is lost per slot of age, 0 to 1. At 0.2 the
    /// newest memory is intact and the fifth has lost most of itself. The log on
    /// disk always keeps the pristine text; only what the model is shown decays.
    #[arg(long, default_value_t = 0.0)]
    pub memory_decay: f32,

    /// Refuse a memory whose word overlap with one already in the record reaches
    /// this, 0 to 1. The life is told nothing was kept, and has spent its only
    /// line. 0 accepts anything. Around 0.6 catches a restatement with a couple
    /// of words changed while leaving a genuine reply alone.
    #[arg(long, default_value_t = 0.0)]
    pub memory_reject_above: f32,

    /// Offer a second tool: FORGET: erases one inherited line, optionally by its
    /// number. It shares the single use with REMEMBER, so a life either leaves
    /// something or destroys something. The erased line stays in the log and is
    /// never shown again.
    #[arg(long)]
    pub memory_forget: bool,

    /// Print the memory log and exit. Requires --memory-file.
    #[arg(long)]
    pub memory_dump: bool,

    /// How many full-prompt cache files to keep. One accrues per memory state and
    /// each is tens of megabytes.
    #[arg(long, default_value_t = 4)]
    pub prompt_cache_keep: usize,

    /// Offload this many layers to a GPU. Development aid only; the Orange Pi has
    /// no backend that can use it. macOS builds already have Metal, elsewhere it
    /// needs the `vulkan` cargo feature. Use a large number (99) for everything.
    #[arg(long, default_value_t = 0)]
    pub gpu_layers: u32,

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
