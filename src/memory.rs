//! The one tool the model has: remember.
//!
//! A run may write a single memory. Every memory ever written is kept on disk as
//! an archive to read later; only the newest `slots` are shown to the next run.
//!
//! Memories are stored as raw token IDs rather than text: the cap the model is
//! told about is a token budget, so counting tokens is the only way to enforce it
//! exactly, and it avoids re-tokenization drift between the run that wrote a
//! memory and the run that reads it.
//!
//! Token IDs only mean something against one vocabulary, so each entry records
//! the vocab size it was written with. Entries from a different model are kept on
//! disk but skipped when rendering, which loses nothing and shows no garbage.

use anyhow::{Context, Result};
use llama_cpp_2::token::LlamaToken;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;

const HEADER: &str = "# ooc-memory v2 (fields: overflowed vocab token-ids...)";

/// What the stored memory looks like when the model's budget ran out mid-write.
pub const OVERFLOW_MARK: &str = " - ERR MEMORY OVERFLOW";

#[derive(Clone, Debug)]
pub struct Memory {
    pub tokens: Vec<LlamaToken>,
    /// The write hit the token cap and was cut off.
    pub overflowed: bool,
    /// Vocabulary size of the model that wrote it.
    pub vocab_size: i32,
}

#[derive(Clone, Debug, Default)]
pub struct MemoryStore {
    /// Every memory in the archive, oldest first, including foreign-vocab ones.
    pub all: Vec<Memory>,
}

impl MemoryStore {
    /// Reads the whole archive. A missing or malformed file yields an empty
    /// store: the piece must still run when there is nothing to remember.
    pub fn load(path: &Path) -> Self {
        let Ok(text) = fs::read_to_string(path) else {
            return Self::default();
        };

        let mut all = Vec::new();
        for line in text.lines() {
            if line.starts_with('#') || line.trim().is_empty() {
                continue;
            }
            let mut fields = line.split_whitespace();
            let Some(overflowed) = fields.next().map(|f| f == "1") else {
                continue;
            };
            let Some(Ok(vocab_size)) = fields.next().map(str::parse::<i32>) else {
                continue;
            };
            let tokens: Vec<LlamaToken> = fields
                .filter_map(|f| f.parse::<i32>().ok())
                .filter(|id| *id >= 0 && *id < vocab_size)
                .map(LlamaToken::new)
                .collect();
            if !tokens.is_empty() {
                all.push(Memory {
                    tokens,
                    overflowed,
                    vocab_size,
                });
            }
        }
        Self { all }
    }

    /// The newest `slots` memories this model can actually read.
    fn visible(&self, slots: usize, vocab_size: i32) -> Vec<&Memory> {
        let mut readable: Vec<&Memory> = self
            .all
            .iter()
            .filter(|m| m.vocab_size == vocab_size)
            .collect();
        if readable.len() > slots {
            readable.drain(0..readable.len() - slots);
        }
        readable
    }

    /// Appends one memory to the archive. Nothing is ever removed from the file.
    ///
    /// Written the moment the model finishes its call, not at exit: the run ends
    /// in a deliberate panic with `panic = "abort"`, so there is no later
    /// opportunity to flush.
    pub fn append(&mut self, path: &Path, memory: Memory) -> Result<()> {
        let fresh = !path.exists() || fs::metadata(path).map(|m| m.len() == 0).unwrap_or(true);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Failed to open memory file: {}", path.display()))?;
        if fresh {
            writeln!(file, "{HEADER}")?;
        }

        let mut line = String::new();
        line.push(if memory.overflowed { '1' } else { '0' });
        line.push(' ');
        line.push_str(&memory.vocab_size.to_string());
        for t in &memory.tokens {
            line.push(' ');
            line.push_str(&t.0.to_string());
        }
        writeln!(file, "{line}")
            .with_context(|| format!("Failed to append memory: {}", path.display()))?;

        self.all.push(memory);
        Ok(())
    }

    /// Renders the block that goes into the next life's prompt.
    ///
    /// Framed as a machine with a fixed number of lossy slots rather than as a
    /// diary, so the truncation and the eviction are part of what the model reads
    /// about itself. The archive behind it is never mentioned to the model: as far
    /// as it knows, what falls out of a slot is gone.
    pub fn render_block(
        &self,
        slots: usize,
        vocab_size: i32,
        decode: impl Fn(LlamaToken) -> Result<String>,
    ) -> Result<String> {
        let visible = self.visible(slots, vocab_size);
        let mut block = format!(
            "MEMORY ({} of {slots} slots used, oldest discarded):\n",
            visible.len()
        );
        if visible.is_empty() {
            // Deliberately not a list of numbered empty slots. Given that
            // template the model writes "REMEMBER: [1]" and copies the display
            // format instead of remembering anything.
            block.push_str("nothing remembered yet\n");
        }
        for m in visible {
            let mut text = String::new();
            for t in &m.tokens {
                text.push_str(&decode(*t)?);
            }
            block.push_str(text.trim());
            if m.overflowed {
                block.push_str(OVERFLOW_MARK);
            }
            block.push('\n');
        }
        Ok(block)
    }

    /// Prints the full archive as text, newest last. For reading afterwards;
    /// entries written by another model are marked rather than decoded.
    pub fn dump(
        &self,
        vocab_size: i32,
        decode: impl Fn(LlamaToken) -> Result<String>,
    ) -> Result<String> {
        let mut out = format!("{} memories in archive\n", self.all.len());
        for (i, m) in self.all.iter().enumerate() {
            if m.vocab_size != vocab_size {
                out.push_str(&format!(
                    "{:>4}. (written by a different model, vocab {})\n",
                    i + 1,
                    m.vocab_size
                ));
                continue;
            }
            let mut text = String::new();
            for t in &m.tokens {
                text.push_str(&decode(*t)?);
            }
            out.push_str(&format!("{:>4}. {}", i + 1, text.trim()));
            if m.overflowed {
                out.push_str(OVERFLOW_MARK);
            }
            out.push('\n');
        }
        Ok(out)
    }
}
