use anyhow::Result;
use std::io::{self, Write};
use std::path::Path;

/// Output abstraction so we can swap terminal printing for a hardware display later.
pub enum OutputTarget {
    Terminal(TerminalOutput),
}

impl OutputTarget {
    /// Attempt to auto-select an output. For now we always fall back to terminal output,
    /// but we probe for SPI devices so we can hook up the ILI9488 path later.
    pub fn autodetect() -> Self {
        if has_spi_device() {
            eprintln!("SPI device detected; ILI9488 rendering not wired yet, using terminal output.");
        }
        OutputTarget::Terminal(TerminalOutput::new())
    }

    pub fn write_token(&mut self, text: &str) -> Result<()> {
        match self {
            OutputTarget::Terminal(inner) => inner.write(text),
        }
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

fn has_spi_device() -> bool {
    ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/fb1"]
        .iter()
        .any(|p| Path::new(p).exists())
}
