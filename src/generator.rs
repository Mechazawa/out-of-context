use anyhow::{Context, Result};
use llama_cpp_2::context::LlamaContext;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use crate::llm::{LLMSetup, LlamaBatchWrapper};

/// Generates text infinitely until the context window is exhausted
pub fn generate_infinite(llm_setup: &LLMSetup, context: &mut LlamaContext, prompt_file: &Path, context_size: usize) -> Result<()> {
    // Read system prompt from file
    let system_prompt = fs::read_to_string(prompt_file)
        .with_context(|| format!("Failed to read prompt file: {}", prompt_file.display()))?;

    println!("\n=== System Prompt ===");
    println!("{}", system_prompt.trim());
    println!("=== Beginning Generation ===\n");

    // Tokenize the system prompt
    let prompt_tokens = llm_setup.tokenize(&system_prompt, true)?;
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

    // Import for sampling
    use llama_cpp_2::token::data_array::LlamaTokenDataArray;

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

        // Simple greedy sampling (take the most likely token)
        let next_token = token_data_array.sample_token_greedy();

        // Decode token to text
        let token_text = llm_setup.decode_token(next_token)?;

        // Print token immediately (streaming output)
        print!("{}", token_text);
        io::stdout().flush()?;

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
