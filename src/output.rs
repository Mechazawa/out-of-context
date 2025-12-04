use anyhow::Result;
use std::io::{self, Write};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

/// Output abstraction so we can swap terminal printing for a hardware display later.
pub struct OutputTarget {
    terminal: TerminalOutput,
    file: Option<FileOutput>,
}

impl OutputTarget {
    /// Attempt to auto-select an output. For now we always fall back to terminal output,
    /// but we probe for SPI devices so we can hook up the ILI9488 path later.
    pub fn autodetect(mirror_file: Option<&PathBuf>) -> Result<Self> {
        if has_spi_device() {
            eprintln!("SPI device detected; ILI9488 rendering not wired yet, using terminal output.");
        }

        let file = if let Some(path) = mirror_file {
            Some(FileOutput::new(path)?)
        } else {
            None
        };

        Ok(OutputTarget {
            terminal: TerminalOutput::new(),
            file,
        })
    }

    pub fn write_token(&mut self, text: &str) -> Result<()> {
        self.terminal.write(text)?;
        if let Some(f) = &mut self.file {
            f.write(text)?;
        }
        Ok(())
    }
}

pub struct TerminalOutput;

impl TerminalOutput {
    pub fn new() -> Self {
        Self
    }

    pub fn write(&mut self, text: &str) -> Result<()> {
        print!("{}", text);
        io::stdout().flush()?;
        Ok(())
    }
}

pub struct FileOutput {
    file: File,
}

impl FileOutput {
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .append(false)
            .truncate(true)
            .open(path)?;

        Ok(Self { file })
    }

    pub fn write(&mut self, text: &str) -> Result<()> {
        self.file.write_all(text.as_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

fn has_spi_device() -> bool {
    ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/fb1"]
        .iter()
        .any(|p| Path::new(p).exists())
}
