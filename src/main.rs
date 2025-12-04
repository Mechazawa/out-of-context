mod cli;
mod generator;
mod llm;
mod model;

use anyhow::Result;
use cli::Args;

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

    // Create context
    let mut context = llm_setup.create_context(args.context_size)?;

    // Start infinite generation
    generator::generate_infinite(&llm_setup, &mut context, &args.prompt_file, args.context_size)?;

    Ok(())
}
