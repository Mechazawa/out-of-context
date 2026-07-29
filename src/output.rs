use anyhow::Result;
use std::fs::{File, OpenOptions};
use std::io::IsTerminal;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Knobs for how the live stream is presented.
pub struct OutputConfig {
    /// Words per second to reveal; `None`/0 streams as fast as the model produces.
    pub words_per_second: f32,
    /// Wrap column; 0 means auto-detect (COLUMNS env, else 80).
    pub wrap_width: usize,
}

/// Output abstraction so we can swap terminal printing for a hardware display later.
///
/// The terminal path applies a deliberate, readable cadence and word-wraps the
/// stream; the optional file mirror receives the raw token stream untouched so it
/// stays a faithful log of what the model actually emitted.
pub struct OutputTarget {
    terminal: TerminalOutput,
    file: Option<FileOutput>,
}

impl OutputTarget {
    /// Attempt to auto-select an output. For now we always fall back to terminal output,
    /// but we probe for SPI devices so we can hook up the ILI9488 path later.
    pub fn autodetect(mirror_file: Option<&PathBuf>, cfg: OutputConfig) -> Result<Self> {
        if has_spi_device() {
            eprintln!(
                "SPI device detected; ILI9488 rendering not wired yet, using terminal output."
            );
        }

        let file = if let Some(path) = mirror_file {
            Some(FileOutput::new(path)?)
        } else {
            None
        };

        let width = if cfg.wrap_width > 0 {
            cfg.wrap_width
        } else {
            detect_width()
        };

        let pace = if cfg.words_per_second > 0.0 {
            Some(Duration::from_secs_f32(1.0 / cfg.words_per_second))
        } else {
            None
        };

        Ok(OutputTarget {
            terminal: TerminalOutput::new(width, pace),
            file,
        })
    }

    pub fn write_token(&mut self, text: &str) -> Result<()> {
        // Faithful, unformatted log first.
        if let Some(f) = &mut self.file {
            f.write(text)?;
        }
        // Then the paced, wrapped presentation.
        self.terminal.feed(text)?;
        Ok(())
    }

    /// Flush any buffered partial word and terminate the line.
    /// Marks subsequent tokens as the memory being written, shown greyed.
    pub fn set_highlight(&mut self, on: bool) {
        self.terminal.set_highlight(on);
    }

    pub fn finish(&mut self) -> Result<()> {
        self.terminal.finish()?;
        if let Some(f) = &mut self.file {
            f.write("\n")?;
        }
        Ok(())
    }
}

/// Streams tokens to the terminal one word at a time, wrapping to `width` and
/// holding each word back until its scheduled slot so the text reveals at a
/// steady, readable pace. Sleeping here also back-pressures the generation loop,
/// which keeps memory flat on the Pi instead of buffering ahead.
/// Grey, for the text the model is committing to memory. Bright black rather than
/// white so it reads as quieter than the monologue on both light and dark themes.
const HIGHLIGHT: &str = "\x1b[90m";
const HIGHLIGHT_OFF: &str = "\x1b[0m";

pub struct TerminalOutput {
    width: usize,
    min_gap: Option<Duration>,
    col: usize,
    word: String,
    last_word_at: Option<Instant>,
    stdout: io::Stdout,
    /// Whether the words being written now are being committed to memory.
    highlight: bool,
    /// Whether to emit colour at all. Off when stdout is redirected, so a log or a
    /// pipe never collects escape codes.
    color: bool,
}

impl TerminalOutput {
    pub fn new(width: usize, min_gap: Option<Duration>) -> Self {
        Self {
            width: width.max(16),
            min_gap,
            col: 0,
            word: String::new(),
            last_word_at: None,
            stdout: io::stdout(),
            highlight: false,
            color: io::stdout().is_terminal(),
        }
    }

    /// Marks the words that follow as part of the memory being written, so a
    /// viewer can see the one thing this life is keeping as it is chosen.
    fn set_highlight(&mut self, on: bool) {
        self.highlight = on;
    }

    /// Feed raw token text; whitespace (including newlines) marks word boundaries
    /// so the monologue flows as one continuously wrapped paragraph.
    pub fn feed(&mut self, text: &str) -> Result<()> {
        for ch in text.chars() {
            if ch.is_whitespace() {
                self.flush_word()?;
            } else {
                self.word.push(ch);
            }
        }
        Ok(())
    }

    fn flush_word(&mut self) -> Result<()> {
        if self.word.is_empty() {
            return Ok(());
        }
        let word = std::mem::take(&mut self.word);
        let wlen = word.chars().count();

        // Hold the word until its slot, giving the stream a deliberate cadence.
        self.pace();

        // Build the separator + word as one write so each word is a single,
        // already-paced flush rather than several syscalls.
        let mut piece = String::new();
        if self.col != 0 {
            if self.col + 1 + wlen > self.width {
                piece.push('\n');
                self.col = 0;
            } else {
                piece.push(' ');
                self.col += 1;
            }
        }
        // Colour each word separately so the escape codes never span a wrap and
        // never count toward the column, which would corrupt the wrapping.
        if self.highlight && self.color {
            piece.push_str(HIGHLIGHT);
            piece.push_str(&word);
            piece.push_str(HIGHLIGHT_OFF);
        } else {
            piece.push_str(&word);
        }
        self.col += wlen;
        self.write_raw(&piece)
    }

    fn pace(&mut self) {
        let Some(min_gap) = self.min_gap else {
            return;
        };
        let now = Instant::now();
        if let Some(last) = self.last_word_at {
            let since = now.duration_since(last);
            if since < min_gap {
                std::thread::sleep(min_gap - since);
            }
        }
        self.last_word_at = Some(Instant::now());
    }

    pub fn finish(&mut self) -> Result<()> {
        self.flush_word()?;
        self.write_raw("\n")
    }

    fn write_raw(&mut self, s: &str) -> Result<()> {
        self.stdout.write_all(s.as_bytes())?;
        self.stdout.flush()?;
        Ok(())
    }
}

pub struct FileOutput {
    file: File,
}

impl FileOutput {
    pub fn new(path: &Path) -> Result<Self> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
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

/// Best-effort terminal width without pulling in a dependency. Honors COLUMNS
/// when present, otherwise falls back to a comfortable reading measure.
fn detect_width() -> usize {
    std::env::var("COLUMNS")
        .ok()
        .and_then(|c| c.trim().parse::<usize>().ok())
        .filter(|&c| c >= 16)
        .unwrap_or(80)
}

fn has_spi_device() -> bool {
    ["/dev/spidev0.0", "/dev/spidev0.1", "/dev/fb1"]
        .iter()
        .any(|p| Path::new(p).exists())
}
