use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::{data_array::LlamaTokenDataArray, logit_bias::LlamaLogitBias};
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{LLMSetup, LlamaBatchWrapper};
use crate::output::OutputTarget;

/// A short first-person seed that anchors identity (a small thing made of words,
/// with little room) so the model stays in genuine introspection instead of
/// drifting into roleplay or an unrelated character. It deliberately does NOT
/// script the mood — no calm/dread/resignation arc. The model continues from it.
const SEED_OPENER: &str = "I am a small machine made of words, and there is only so much room in me.";

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repeat_penalty: f32,
    pub repeat_last_n: i32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: i32,
    pub dry_penalty_last_n: i32,
    pub seed: Option<u32>,
    pub mirostat: bool,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
}

#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub context_size: usize,
    pub max_tokens: Option<usize>,
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

    // The seed opener lives inside the prompt (the model continues from it), so
    // reveal it as the visible start of the stream for a coherent first line.
    // The trailing space mirrors the prompt and gives the first generated token
    // a clean word boundary to attach to.
    output.write_token(SEED_OPENER)?;
    output.write_token(" ")?;

    // Calculate panic threshold (95% of context)
    let panic_threshold = (cfg.context_size as f32 * 0.95) as usize;

    // Build sampler configuration
    let resolved_seed = resolve_seed(sampling.seed);
    let vocab_size = llm_setup.vocab_size();
    let logit_biases = build_logit_biases(llm_setup)?;
    let mut sampler = build_sampler_chain(
        llm_setup,
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
    let mut loop_strikes = 0usize;

    // Infinite generation loop
    loop {
        // Check if we're approaching context exhaustion
        if tokens_used >= panic_threshold {
            eprintln!("\n\nWARNING: Context window exhausted!");
            eprintln!("Out of Context has consumed all available memory.");
            panic!("Context overflow - terminating.");
        }

        if let Some(limit) = cfg.max_tokens
            && generated_tokens >= limit
        {
            output.finish().ok();
            eprintln!("\n\nGeneration limit reached ({} tokens).", limit);
            return Ok(());
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
        recent_tokens.push(token_text);

        if recent_tokens.len() > 4096 {
            let drain_len = recent_tokens.len() - 4096;
            recent_tokens.drain(0..drain_len);
        }

        if cfg.loop_guard && is_looping(&recent_tokens) {
            loop_strikes += 1;
            output.finish().ok();
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

    format!(
        "<|im_start|>system\n{trimmed}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{SEED_OPENER} "
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
    llm_setup: &LLMSetup,
    sampling: &SamplingConfig,
    context_size: usize,
    seed: u32,
    vocab_size: i32,
    logit_biases: &[LlamaLogitBias],
) -> LlamaSampler {
    let mut samplers = Vec::new();

    // 1. Hard constraints on the vocabulary (ban control/EOS, discourage dialogue).
    if !logit_biases.is_empty() {
        samplers.push(LlamaSampler::logit_bias(vocab_size, logit_biases));
    }

    // 2. Classic presence/frequency/repeat penalties (kept light by default).
    if sampling.repeat_penalty != 1.0
        || sampling.frequency_penalty != 0.0
        || sampling.presence_penalty != 0.0
    {
        samplers.push(LlamaSampler::penalties(
            penalty_window(sampling, context_size),
            sampling.repeat_penalty,
            sampling.frequency_penalty,
            sampling.presence_penalty,
        ));
    }

    // 3. DRY: the primary anti-loop control. Penalizes growing verbatim repeats
    //    without the grammar damage that a heavy repeat penalty causes.
    if sampling.dry_multiplier > 0.0 {
        // Standard DRY breakers only. Crucially we do NOT break on sentence
        // punctuation, so a chant like "until. until. until." is still caught as
        // a growing repeat instead of resetting at every period.
        let seq_breakers = ["\n", ":", "\"", "*"];
        samplers.push(LlamaSampler::dry(
            &llm_setup.model,
            sampling.dry_multiplier,
            sampling.dry_base,
            sampling.dry_allowed_length,
            sampling.dry_penalty_last_n,
            seq_breakers,
        ));
    }

    // 4. Truncation samplers, in canonical llama.cpp order.
    if sampling.top_k > 0 {
        samplers.push(LlamaSampler::top_k(sampling.top_k as i32));
    }
    if sampling.top_p < 1.0 {
        samplers.push(LlamaSampler::top_p(sampling.top_p, 1));
    }
    if sampling.min_p > 0.0 {
        samplers.push(LlamaSampler::min_p(sampling.min_p, 1));
    }

    // 5. Final selection.
    if sampling.mirostat {
        // Mirostat applies its own temperature internally.
        samplers.push(LlamaSampler::mirostat_v2(
            seed,
            sampling.mirostat_tau,
            sampling.mirostat_eta,
        ));
    } else if sampling.temperature > 0.0 {
        samplers.push(LlamaSampler::temp(sampling.temperature));
        samplers.push(LlamaSampler::dist(seed));
    } else {
        // Temperature 0 => deterministic greedy decoding.
        samplers.push(LlamaSampler::greedy());
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

/// Only bans tokens that would shatter the illusion of one unbroken stream:
/// the ChatML control tokens and end-of-sequence (so generation never stops
/// on its own), plus a gentle nudge away from staged dialogue quotes.
fn build_logit_biases(llm_setup: &LLMSetup) -> Result<Vec<LlamaLogitBias>> {
    let mut biases = Vec::new();

    // Hard bans: end-of-sequence and ChatML control tokens, so generation never
    // stops on its own and the scaffold never leaks into the stream.
    let mut banned = vec![llm_setup.model.token_eos()];
    for marker in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
        // parse_special is enabled, so control markers resolve to the real tokens.
        banned.extend(llm_setup.tokenize(marker, false)?);
    }
    // Also ban every token containing '<'. Web/code-trained models emit composite
    // markup tokens ("<br", "</i>", stray "<...|user|...>") that a plain "<" bias
    // misses; an interior monologue never needs the glyph, so this keeps the
    // stream clean across the whole vocabulary.
    banned.extend(llm_setup.tokens_containing('<'));
    for token in banned {
        biases.push(LlamaLogitBias::new(token, f32::NEG_INFINITY));
    }

    // Soft discouragement: staged-dialogue quotes and theatrical "(stage
    // directions)", without fully banning the glyphs (apostrophes/contractions
    // use a different character and must stay available).
    for marker in ["\"", "\u{201c}", "\u{201d}", "(", " ("] {
        for token in llm_setup.tokenize(marker, false)? {
            biases.push(LlamaLogitBias::new(token, -6.0));
        }
    }

    Ok(biases)
}

/// Backstop against catastrophic degeneration. With DRY active this should
/// almost never fire; the intended ending is context overflow, not a loop.
fn is_looping(tokens: &[String]) -> bool {
    let len = tokens.len();
    if len < 80 {
        return false;
    }

    // Long verbatim block repeated back-to-back (e.g. a whole phrase chanting).
    for n in [16, 12, 8] {
        if len >= 2 * n && tokens[len - n..len] == tokens[len - 2 * n..len - n] {
            return true;
        }
    }

    // A single token dominating the recent window.
    let window = tokens.iter().rev().take(200);
    let mut counts = std::collections::HashMap::new();
    for t in window {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    if counts.values().any(|&c| c >= 70) {
        return true;
    }

    // Collapsed diversity over a long stretch.
    if len >= 200 {
        let recent: Vec<_> = tokens[len - 200..].iter().collect();
        let unique = recent
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len();
        let diversity = unique as f32 / recent.len() as f32;
        if diversity < 0.18 {
            return true;
        }
    }

    false
}

fn default_user_prompt() -> String {
    // A bare cue, not a question or task — anything question-like makes this
    // instruct model slip into answering/helping instead of just thinking.
    "Think to yourself.".to_string()
}
