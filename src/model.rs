use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::cmp::min;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Resolves the model path and ensures it exists
///
/// If `model_spec` is a URL, downloads to `model_dir` and returns the local path.
/// If `model_spec` is a local path, verifies it exists and returns it.
pub async fn resolve_model(model_spec: &str, model_dir: &Path) -> Result<PathBuf> {
    // Check if model_spec is a URL
    if model_spec.starts_with("http://") || model_spec.starts_with("https://") {
        // Extract filename from URL
        let filename = model_spec
            .rsplit('/')
            .next()
            .context("Invalid model URL: no filename")?;

        let model_path = model_dir.join(filename);

        // Check if already downloaded
        if model_path.exists() {
            println!("Model found at: {}", model_path.display());
            return Ok(model_path);
        }

        println!("Model not found locally");
        println!("Downloading from: {}", model_spec);

        // Create model directory if it doesn't exist
        std::fs::create_dir_all(model_dir)
            .with_context(|| format!("Failed to create directory: {}", model_dir.display()))?;

        // Download the model
        download_model(model_spec, &model_path).await?;

        Ok(model_path)
    } else {
        // Treat as local file path
        let model_path = PathBuf::from(model_spec);

        if !model_path.exists() {
            anyhow::bail!("Model file not found: {}", model_path.display());
        }

        println!("Using local model: {}", model_path.display());
        Ok(model_path)
    }
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
    pb.set_message(format!(
        "Downloading {}",
        destination.file_name().unwrap().to_string_lossy()
    ));

    // Create output file
    let mut file = File::create(destination)
        .with_context(|| format!("Failed to create file: {}", destination.display()))?;

    // Stream download with progress
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.context("Failed to read chunk")?;
        file.write_all(&chunk).context("Failed to write to file")?;

        let new = min(downloaded + (chunk.len() as u64), total_size);
        downloaded = new;
        pb.set_position(new);
    }

    pb.finish_with_message(format!(
        "Downloaded {}",
        destination.file_name().unwrap().to_string_lossy()
    ));
    println!("Model downloaded successfully!");

    Ok(())
}
