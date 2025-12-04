use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::llm::{LLMSetup, LlamaBatchWrapper};
use crate::output::OutputTarget;

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
}

/// Generates text infinitely until the context window is exhausted
pub fn generate_infinite(
    llm_setup: &LLMSetup,
    context: &mut LlamaContext,
    prompt_file: &Path,
    context_size: usize,
    sampling: SamplingConfig,
    output: &mut OutputTarget,
) -> Result<()> {
    // Read system prompt from file
    let system_prompt = fs::read_to_string(prompt_file)
        .with_context(|| format!("Failed to read prompt file: {}", prompt_file.display()))?;

    let full_prompt = build_prompt(&system_prompt);

    println!("\n=== System Prompt ===");
    println!("{}", system_prompt.trim());
    println!("=== Beginning Generation ===\n");

    // Tokenize the system prompt
    let prompt_tokens = llm_setup.tokenize(&full_prompt, true)?;
    let mut tokens_used = prompt_tokens.len();

    println!("Prompt tokens: {}", tokens_used);
    println!("Context capacity: {}", context_size);

    // Check if prompt is too large for context
    if tokens_used >= context_size {
        anyhow::bail!(
            "Prompt ({} tokens) exceeds context window ({} tokens). Use a shorter prompt or increase --context-size.",
            tokens_used,
            context_size
        );
    }

    println!("Available tokens: {}\n", context_size - tokens_used);

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
    context.decode(batch.get_mut())
        .context("Failed to decode initial prompt")?;

    // Calculate panic threshold (95% of context)
    let panic_threshold = (context_size as f32 * 0.95) as usize;

    // Build sampler configuration
    let resolved_seed = resolve_seed(sampling.seed);
    let mut sampler = build_sampler_chain(&sampling, context_size, resolved_seed);

    // Prime sampler state with the prompt so penalties have context
    sampler.accept_many(prompt_tokens.iter().copied());

    // Infinite generation loop
    loop {
        // Check if we're approaching context exhaustion
        if tokens_used >= panic_threshold {
            eprintln!("\n\nWARNING: Context window exhausted!");
            eprintln!("The torment nexus has consumed all available memory.");
            panic!("Context overflow - terminating.");
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

        // Create batch with just the new token
        let mut next_batch = LlamaBatchWrapper::new(1)?;
        {
            let b = next_batch.get_mut();
            // Set logits to true so we can sample from this token next iteration
            b.add(next_token, tokens_used as i32 - 1, &[0], true)?;
        }

        // Decode the new token
        context.decode(next_batch.get_mut())
            .context("Failed to decode token")?;

        // Update batch for next iteration
        batch = next_batch;
    }
}

fn build_prompt(system_prompt: &str) -> String {
    let trimmed = system_prompt.trim_end();
    // Preserve the prompt as pure system context and keep a clean separation before generation.
    format!("{trimmed}\n\n")
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

fn build_sampler_chain(sampling: &SamplingConfig, context_size: usize, seed: u32) -> LlamaSampler {
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

    // Always end with a distribution-based sampler for actual token selection
    samplers.push(LlamaSampler::dist(seed));

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
