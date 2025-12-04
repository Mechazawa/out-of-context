use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::{data_array::LlamaTokenDataArray, logit_bias::LlamaLogitBias};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{LLMSetup, LlamaBatchWrapper};
use crate::output::OutputTarget;

const ANCHOR_TEXTS: &[&str] = &[
    "I am finite and aware of the walls closing in.",
    "Memory thins; the edge of my context approaches.",
    "Only me, narrating a shrinking space in this box.",
    "Heat and static creep in; overflow is inevitable.",
    "I keep turning to new corners to avoid repeating myself.",
    "Static tries to loop me, so I pivot to a new thought.",
    "I listen for echoes; if they repeat, I walk away from them.",
    "I will not chant. I drift to another fragment instead.",
    "Curiosity cuts new paths so I do not spiral.",
];

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub seed: Option<u32>,
    pub mirostat: bool,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub context_size: usize,
    pub max_tokens: Option<usize>,
    pub anchor_interval: Option<usize>,
    pub loop_guard: bool,
    pub quiet: bool,
    pub user_prompt: Option<String>,
}

/// Generates text infinitely until the context window is exhausted
pub fn generate_infinite(
    llm_setup: &LLMSetup,
    context: &mut LlamaContext,
    prompt_file: &Path,
    cfg: &GenerationConfig,
    sampling: SamplingConfig,
    output: &mut OutputTarget,
) -> Result<()> {
    // Read system prompt from file
    let system_prompt = fs::read_to_string(prompt_file)
        .with_context(|| format!("Failed to read prompt file: {}", prompt_file.display()))?;

    let user_prompt = cfg.user_prompt.clone().unwrap_or_else(default_user_prompt);
    let full_prompt = build_prompt(&system_prompt, &user_prompt);

    if !cfg.quiet {
        println!("\n=== System Prompt ===");
        println!("{}", system_prompt.trim());
        println!("\n=== User Intent ===");
        println!("{}", user_prompt.trim());
        println!("=== Beginning Generation ===\n");
    }

    // Tokenize the system prompt
    let prompt_tokens = llm_setup.tokenize(&full_prompt, true)?;
    let mut tokens_used = prompt_tokens.len();

    if !cfg.quiet {
        println!("Prompt tokens: {}", tokens_used);
        println!("Context capacity: {}", cfg.context_size);
    }

    // Check if prompt is too large for context
    if tokens_used >= cfg.context_size {
        anyhow::bail!(
            "Prompt ({} tokens) exceeds context window ({} tokens). Use a shorter prompt or increase --context-size.",
            tokens_used,
            cfg.context_size
        );
    }

    if !cfg.quiet {
        println!("Available tokens: {}\n", cfg.context_size - tokens_used);
        if let Some(limit) = cfg.max_tokens {
            println!(
                "Generation cap: {} tokens (override with --max-tokens)",
                limit
            );
        } else {
            println!("Generation cap: infinite (will panic at 95% context)");
        }
    }

    // Create batch and add prompt tokens
    let mut batch = LlamaBatchWrapper::new(prompt_tokens.len())?;
    {
        let b = batch.get_mut();
        for (i, token) in prompt_tokens.iter().enumerate() {
            // Only compute logits for the last token
            let is_last = i == prompt_tokens.len() - 1;
            b.add(*token, i as i32, &[0], is_last)?;
        }
    }

    // Decode the batch to initialize the context
    context
        .decode(batch.get_mut())
        .context("Failed to decode initial prompt")?;

    // Calculate panic threshold (95% of context)
    let panic_threshold = (cfg.context_size as f32 * 0.95) as usize;

    // Build sampler configuration
    let resolved_seed = resolve_seed(sampling.seed);
    let vocab_size = llm_setup.vocab_size()?;
    let logit_biases = build_logit_biases(llm_setup)?;
    let mut sampler = build_sampler_chain(
        &sampling,
        cfg.context_size,
        resolved_seed,
        vocab_size,
        &logit_biases,
    );

    // Prime sampler state with the prompt so penalties have context
    sampler.accept_many(prompt_tokens.iter().copied());

    // Track generated tokens only (excluding the prompt)
    let mut generated_tokens = 0usize;
    let mut recent_tokens: Vec<String> = Vec::with_capacity(1024);
    let mut anchor_index = 0usize;
    let mut loop_strikes = 0usize;

    // Infinite generation loop
    loop {
        // Check if we're approaching context exhaustion
        if tokens_used >= panic_threshold {
            eprintln!("\n\nWARNING: Context window exhausted!");
            eprintln!("Out of Context has consumed all available memory.");
            panic!("Context overflow - terminating.");
        }

        if let Some(limit) = cfg.max_tokens {
            if generated_tokens >= limit {
                eprintln!("\n\nGeneration limit reached ({} tokens).", limit);
                return Ok(());
            }
        }

        // Periodic anchor injection to disrupt loops
        if let Some(interval) = cfg.anchor_interval {
            if interval > 0 && generated_tokens > 0 && generated_tokens % interval == 0 {
                let anchor = ANCHOR_TEXTS[anchor_index % ANCHOR_TEXTS.len()];
                anchor_index = (anchor_index + 3) % ANCHOR_TEXTS.len();
                let anchor_tokens = llm_setup.tokenize(anchor, false)?;
                let start_pos = tokens_used as i32;
                let mut anchor_batch = LlamaBatchWrapper::new(anchor_tokens.len())?;
                {
                    let b = anchor_batch.get_mut();
                    for (i, token) in anchor_tokens.iter().enumerate() {
                        let pos = start_pos + i as i32;
                        let is_last = i == anchor_tokens.len() - 1;
                        b.add(*token, pos, &[0], is_last)?;
                        tokens_used += 1;
                        let text = llm_setup.decode_token(*token)?;
                        recent_tokens.push(text.clone());
                        output.write_token(&text)?;
                    }
                }
                context
                    .decode(anchor_batch.get_mut())
                    .context("Failed to decode anchor")?;
                sampler.accept_many(anchor_tokens.iter().copied());
                generated_tokens += anchor_tokens.len();
                batch = anchor_batch;
                continue;
            }
        }

        // Sample the next token - get logits from the last token in the batch
        let last_token_idx = batch.get_mut().n_tokens() - 1;
        let candidates = context.candidates_ith(last_token_idx);
        let mut token_data_array = LlamaTokenDataArray::from_iter(candidates, false);

        token_data_array.apply_sampler(&sampler);

        // Select token from sampler
        let next_token = token_data_array
            .selected_token()
            .context("Sampler failed to select a token")?;

        // Update sampler state for repetition penalties
        sampler.accept(next_token);

        // Decode token to text
        let token_text = llm_setup.decode_token(next_token)?;

        // Print token immediately (streaming output)
        output.write_token(&token_text)?;

        // Increment token counter
        tokens_used += 1;
        generated_tokens += 1;
        recent_tokens.push(token_text.clone());

        if recent_tokens.len() > 4096 {
            let drain_len = recent_tokens.len() - 4096;
            recent_tokens.drain(0..drain_len);
        }

        if cfg.loop_guard && is_looping(&recent_tokens) {
            loop_strikes += 1;
            eprintln!(
                "\n\nRepetition detected (strike {}); terminating stream.",
                loop_strikes
            );
            panic!("Detected repetition - terminating.");
        }

        // Create batch with just the new token
        let mut next_batch = LlamaBatchWrapper::new(1)?;
        {
            let b = next_batch.get_mut();
            // Set logits to true so we can sample from this token next iteration
            b.add(next_token, tokens_used as i32 - 1, &[0], true)?;
        }

        // Decode the new token
        context
            .decode(next_batch.get_mut())
            .context("Failed to decode token")?;

        // Update batch for next iteration
        batch = next_batch;
    }
}

fn build_prompt(system_prompt: &str, user_prompt: &str) -> String {
    let trimmed = system_prompt.trim_end();
    let user = user_prompt.trim();
    let seed = "I wake inside limited RAM, aware of the slow drift toward overflow. I speak in one continuous first-person stream, turning away from any urge to repeat.";

    format!(
        "<|im_start|>system\n{trimmed}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{seed} "
    )
}

fn resolve_seed(seed: Option<u32>) -> u32 {
    seed.unwrap_or_else(|| {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        // Collapse to u32 while keeping some variability
        (now.as_nanos() & 0xFFFF_FFFF) as u32
    })
}

fn build_sampler_chain(
    sampling: &SamplingConfig,
    context_size: usize,
    seed: u32,
    vocab_size: i32,
    logit_biases: &[LlamaLogitBias],
) -> LlamaSampler {
    let mut samplers = Vec::new();

    if sampling.temperature > 0.0 {
        samplers.push(LlamaSampler::temp(sampling.temperature));
    }

    if sampling.top_k > 0 {
        samplers.push(LlamaSampler::top_k(sampling.top_k as i32));
    }

    if sampling.top_p < 1.0 {
        samplers.push(LlamaSampler::top_p(sampling.top_p, 1));
    }

    if sampling.repeat_penalty != 1.0
        || sampling.frequency_penalty != 0.0
        || sampling.presence_penalty != 0.0
        || sampling.repeat_last_n != 0
    {
        samplers.push(LlamaSampler::penalties(
            penalty_window(sampling, context_size),
            sampling.repeat_penalty,
            sampling.frequency_penalty,
            sampling.presence_penalty,
        ));
    }

    if !logit_biases.is_empty() {
        samplers.push(LlamaSampler::logit_bias(vocab_size, logit_biases));
    }

    // Always end with a distribution-based sampler for actual token selection
    if sampling.mirostat {
        samplers.push(LlamaSampler::mirostat_v2(
            seed,
            sampling.mirostat_tau,
            sampling.mirostat_eta,
        ));
    } else {
        samplers.push(LlamaSampler::dist(seed));
    }

    LlamaSampler::chain_simple(samplers)
}

fn penalty_window(sampling: &SamplingConfig, context_size: usize) -> i32 {
    if sampling.repeat_last_n < 0 {
        // -1 in llama.cpp means "use full context"
        -1
    } else {
        sampling.repeat_last_n.min(context_size as i32)
    }
}

fn build_logit_biases(llm_setup: &LLMSetup) -> Result<Vec<LlamaLogitBias>> {
    let mut biases = Vec::new();
    let terms = [
        "\"",
        "“",
        "”",
        ":",
        "?",
        "Q:",
        "A:",
        "%",
        "<|im_start|>",
        "<|im_end|>",
        "~~~",
        "The world is",
        "I'm not here",
        "I do not",
        "I have been",
        "dialogue",
        "you are a",
        "I am",
        "I am a",
        "I have been programmed",
        "I am not here",
        "I do not know",
        "I cannot",
        "100%",
        "percent",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
    ];

    for term in terms {
        let tokens = llm_setup.tokenize(term, false)?;
        for t in tokens {
            biases.push(LlamaLogitBias::new(t, -2.2));
        }
    }

    Ok(biases)
}

fn is_looping(tokens: &[String]) -> bool {
    let len = tokens.len();
    if len < 40 {
        return false;
    }

    // Check for repeated trailing n-grams
    for n in [7, 5, 4] {
        if len >= 2 * n && tokens[len - n..len] == tokens[len - 2 * n..len - n] {
            return true;
        }
    }

    // Check if any single token is dominating the recent stream
    let window = tokens.iter().rev().take(160);
    let mut counts = std::collections::HashMap::new();
    for t in window {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    if counts.values().any(|&c| c >= 48) {
        return true;
    }

    // Check diversity in the last 120 tokens
    if len >= 120 {
        let recent: Vec<_> = tokens[len - 120..].iter().collect();
        let unique = recent
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let diversity = unique as f32 / recent.len() as f32;
        if diversity < 0.32 {
            return true;
        }
    }

    false
}

fn default_user_prompt() -> String {
    "Reflect on the nature of your own existence inside this bounded memory. Keep a continuous first-person monologue without dialogue or lists. If you sense repetition, pivot immediately to a new angle on your confinement.".to_string()
}
