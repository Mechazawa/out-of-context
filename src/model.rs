use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::min;
use std::fs::File;
use std::io::Write;
use std::path::Path;

const DEFAULT_MODEL_URL: &str =
    "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/main/SmolLM2-135M-Instruct-Q4_K_M.gguf";

/// Ensures the model file exists, downloading it if necessary
pub async fn ensure_model_exists(model_path: &Path) -> Result<()> {
    // Check if model already exists
    if model_path.exists() {
        println!("Model found at: {}", model_path.display());
        return Ok(());
    }

    println!("Model not found at: {}", model_path.display());
    println!("Downloading from Hugging Face...");

    // Create parent directory if it doesn't exist
    if let Some(parent) = model_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create directory: {}", parent.display()))?;
    }

    download_model(DEFAULT_MODEL_URL, model_path).await?;

    Ok(())
}

/// Downloads a model from a URL with progress bar
async fn download_model(url: &str, destination: &Path) -> Result<()> {
    // Create HTTP client
    let client = reqwest::Client::new();

    // Send GET request
    let response = client
        .get(url)
        .send()
        .await
        .context("Failed to send download request")?;

    // Check if request was successful
    if !response.status().is_success() {
        anyhow::bail!("Failed to download model: HTTP {}", response.status());
    }

    // Get content length for progress bar
    let total_size = response.content_length().unwrap_or(0);

    // Create progress bar
    let pb = ProgressBar::new(total_size);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{msg}\n{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {bytes}/{total_bytes} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    pb.set_message(format!("Downloading {}", destination.file_name().unwrap().to_string_lossy()));

    // Create output file
    let mut file = File::create(destination)
        .with_context(|| format!("Failed to create file: {}", destination.display()))?;

    // Stream download with progress
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk)
            .context("Failed to write to file")?;

        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!("Downloaded {}", destination.file_name().unwrap().to_string_lossy()));
    println!("Model downloaded successfully!");

    Ok(())
}
