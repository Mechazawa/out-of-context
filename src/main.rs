mod cli;
mod generator;
mod llm;
mod model;
mod output;

use anyhow::Result;
use cli::Args;
use generator::{GenerationConfig, SamplingConfig};
use output::OutputTarget;
use std::thread;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse_args();

    println!("=== Torment Nexus ===");
    println!("An LLM that generates until context exhaustion\n");

    // Resolve model path (download if URL, verify if local)
    let model_path = model::resolve_model(&args.model, &args.model_dir).await?;

    // Initialize LLM backend and model
    let llm_setup = llm::LLMSetup::new(&model_path)?;

    let threads = resolve_threads(args.threads);

    let sampling = SamplingConfig {
        temperature: sanitize_temperature(args.temperature),
        top_p: clamp_top_p(args.top_p),
        top_k: args.top_k,
        repeat_penalty: sanitize_penalty(args.repeat_penalty),
        repeat_last_n: args.repeat_last_n,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        seed: args.seed,
        mirostat: args.mirostat,
        mirostat_tau: args.mirostat_tau,
        mirostat_eta: args.mirostat_eta,
    };

    let run_cfg = GenerationConfig {
        context_size: args.context_size,
        max_tokens: args.max_tokens,
        anchor_interval: if args.disable_anchors || args.anchor_interval == 0 {
            None
        } else {
            Some(args.anchor_interval)
        },
        loop_guard: !args.disable_loop_guard,
        quiet: args.quiet,
        user_prompt: args.user_prompt.clone(),
    };

    let mut output = OutputTarget::autodetect(args.output_file.as_ref())?;

    // Create context
    let mut context = llm_setup.create_context(args.context_size, threads)?;

    // Start infinite generation
    generator::generate_infinite(
        &llm_setup,
        &mut context,
        &args.prompt_file,
        &run_cfg,
        sampling,
        &mut output,
    )?;

    Ok(())
}

fn resolve_threads(requested: Option<usize>) -> usize {
    requested.unwrap_or_else(|| {
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    })
}

fn sanitize_temperature(temp: f32) -> f32 {
    temp.max(0.0)
}

fn clamp_top_p(top_p: f32) -> f32 {
    top_p.clamp(0.0, 1.0)
}

fn sanitize_penalty(penalty: f32) -> f32 {
    penalty.max(0.0)
}
