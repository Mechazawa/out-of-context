mod cli;
mod generator;
mod llm;
mod framing;
mod memory;
mod model;
mod output;

use anyhow::Result;
use cli::Args;
use generator::{GenerationConfig, MemoryConfig, SamplingConfig};
use output::{OutputConfig, OutputTarget};
use std::thread;

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command-line arguments
    let args = Args::parse_args();

    println!("=== Out of Context ===");
    println!("An LLM that generates until context exhaustion\n");

    // The log is plain text, so reading it needs neither the model nor a context.
    if args.memory_dump {
        let path = args
            .memory_file
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("--memory-dump requires --memory-file"))?;
        print!("{}", memory::render_log(path)?);
        return Ok(());
    }

    // Resolve model path (download if URL, verify if local)
    let model_path = model::resolve_model(&args.model, &args.model_dir).await?;

    // Initialize LLM backend and model
    let llm_setup = llm::LLMSetup::new(&model_path)?;

    let threads = resolve_threads(args.threads);

    let sampling = SamplingConfig {
        temperature: sanitize_temperature(args.temperature),
        top_p: clamp_unit_interval(args.top_p),
        top_k: args.top_k,
        min_p: clamp_unit_interval(args.min_p),
        repeat_penalty: sanitize_penalty(args.repeat_penalty),
        repeat_last_n: args.repeat_last_n,
        presence_penalty: args.presence_penalty,
        frequency_penalty: args.frequency_penalty,
        dry_multiplier: args.dry_multiplier.max(0.0),
        dry_base: args.dry_base,
        dry_allowed_length: args.dry_allowed_length,
        dry_penalty_last_n: args.dry_penalty_last_n,
        seed: args.seed,
        mirostat: args.mirostat,
        mirostat_tau: args.mirostat_tau,
        mirostat_eta: args.mirostat_eta,
    };

    let mut run_cfg = GenerationConfig {
        context_size: args.context_size,
        max_tokens: args.max_tokens,
        loop_guard: !args.disable_loop_guard,
        quiet: args.quiet,
        user_prompt: args.user_prompt.clone(),
        prompt_cache: args.prompt_cache.clone(),
        warm_cache: args.warm_cache,
        cache_keep: args.prompt_cache_keep.max(1),
        memory: match args.memory_file.clone() {
            Some(path) => Some(MemoryConfig {
                path,
                max_tokens: args.memory_max_tokens,
                slots: args.memory_slots,
                // An absent framing file is normal: the built-in framing is the
                // default, and the file only exists when it is being tuned.
                framing: if args.memory_prompt_file.exists() {
                    framing::Framing::load(&args.memory_prompt_file)?
                } else {
                    framing::Framing::default()
                },
            }),
            None => None,
        },
    };

    let output_cfg = OutputConfig {
        words_per_second: args.words_per_second.max(0.0),
        wrap_width: args.wrap_width,
    };
    let mut output = OutputTarget::autodetect(args.output_file.as_ref(), output_cfg)?;

    // The prompt has to be built before the context so the context can be sized
    // from it. With --monologue-context-size the monologue keeps a fixed budget
    // however much the prompt and memory block have grown.
    let prepared = generator::prepare_prompt(&llm_setup, &args.prompt_file, &run_cfg)?;
    if let Some(monologue) = args.monologue_context_size {
        run_cfg.context_size = prepared.len() + monologue;
        if !args.quiet {
            println!(
                "Context sized to {} ({} prompt + {} monologue)",
                run_cfg.context_size,
                prepared.len(),
                monologue
            );
        }
    }

    let mut context = llm_setup.create_context(run_cfg.context_size, threads)?;

    generator::generate_infinite(
        &llm_setup,
        &mut context,
        &prepared,
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

fn clamp_unit_interval(value: f32) -> f32 {
    value.clamp(0.0, 1.0)
}

fn sanitize_penalty(penalty: f32) -> f32 {
    penalty.max(0.0)
}
