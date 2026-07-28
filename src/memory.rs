//! The one tool the model has: remember.
//!
//! A run may write a single memory. Every memory ever written is appended to a
//! plain-text log meant to be read by a human afterwards; only the newest few
//! are shown to the next run.
//!
//! The log is never loaded whole. It is read backwards from the end, far enough
//! to recover the newest entries and the running index, so an installation that
//! has lived thousands of lives costs the same to start as one on its first.

use anyhow::{Context, Result};
use chrono::DateTime;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

const HEADER: &str =
    "# out-of-context memory log. tab-separated: life, unix-time, tokens, status, at-token, text";

/// Appended to what was stored when the model's budget ran out mid-write.
pub const OVERFLOW_MARK: &str = " - ERR MEMORY OVERFLOW";

/// How much of the tail to read per step when walking backwards.
const TAIL_CHUNK: usize = 8 * 1024;

#[derive(Clone, Debug)]
pub struct Memory {
    /// 1-based life number, in write order.
    pub life: u64,
    pub unix_time: u64,
    /// Tokens actually stored, as counted when it was written.
    pub tokens: usize,
    /// The write hit the cap and was cut off.
    pub overflowed: bool,
    /// How many tokens into the monologue the write finished. A write that lands
    /// early can only be made of the inherited block, so this is the diagnostic
    /// for whether a framing delays the decision.
    pub at_token: usize,
    pub text: String,
}

impl Memory {
    fn parse(line: &str) -> Option<Self> {
        let mut f = line.splitn(6, '\t');
        let life = f.next()?.trim().parse().ok()?;
        let unix_time = f.next()?.trim().parse().unwrap_or(0);
        let tokens = f.next()?.trim().parse().unwrap_or(0);
        let overflowed = f.next()?.trim() == "overflow";
        let at_token = f.next()?.trim().parse().unwrap_or(0);
        let text = f.next()?.trim().to_string();
        if text.is_empty() {
            return None;
        }
        Some(Self {
            life,
            unix_time,
            tokens,
            overflowed,
            at_token,
            text,
        })
    }

    fn to_line(&self) -> String {
        // Tabs and newlines would break the one-memory-per-line contract. A
        // memory is captured up to a newline so it cannot contain one, but the
        // model can emit a tab.
        let text: String = self
            .text
            .chars()
            .map(|c| if c.is_control() || c == '\t' { ' ' } else { c })
            .collect();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}",
            self.life,
            self.unix_time,
            self.tokens,
            if self.overflowed { "overflow" } else { "ok" },
            self.at_token,
            text.trim()
        )
    }

    /// What the next life reads: the text, with the truncation visible.
    pub fn display(&self) -> String {
        if self.overflowed {
            format!("{}{}", self.text, OVERFLOW_MARK)
        } else {
            self.text.clone()
        }
    }
}

/// The newest entries plus how many lives have been lived, without reading the
/// whole log.
#[derive(Clone, Debug, Default)]
pub struct MemoryTail {
    /// Oldest first.
    pub recent: Vec<Memory>,
    /// Highest life number seen, so the next life knows its own number.
    pub lives: u64,
}

impl MemoryTail {
    /// Reads the last `want` memories by walking the file backwards.
    ///
    /// A missing or unreadable log yields an empty tail: the piece must still run
    /// on its first life, and a damaged log is indistinguishable from that.
    pub fn load(path: &Path, want: usize) -> Self {
        match Self::try_load(path, want) {
            Ok(tail) => tail,
            Err(_) => Self::default(),
        }
    }

    fn try_load(path: &Path, want: usize) -> Result<Self> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        if len == 0 {
            return Ok(Self::default());
        }

        // Walk backwards in chunks until enough complete lines are in hand. One
        // extra line of slack covers the partial line at the front of a chunk.
        let mut buf: Vec<u8> = Vec::new();
        let mut pos = len;
        let mut memories = loop {
            let step = TAIL_CHUNK.min(pos as usize);
            pos -= step as u64;
            let mut chunk = vec![0u8; step];
            file.seek(SeekFrom::Start(pos))?;
            file.read_exact(&mut chunk)?;
            chunk.extend_from_slice(&buf);
            buf = chunk;

            let text = String::from_utf8_lossy(&buf);
            // Skip the first line unless the file start was reached: it may be a
            // fragment of a longer line that continues before this chunk.
            let complete: Vec<&str> = if pos == 0 {
                text.lines().collect()
            } else {
                text.lines().skip(1).collect()
            };
            let parsed: Vec<Memory> = complete
                .iter()
                .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
                .filter_map(|l| Memory::parse(l))
                .collect();

            if parsed.len() >= want || pos == 0 {
                break parsed;
            }
        };

        let lives = memories.last().map(|m| m.life).unwrap_or(0);
        if memories.len() > want {
            memories.drain(0..memories.len() - want);
        }
        Ok(Self {
            recent: memories,
            lives,
        })
    }

    /// Appends one memory to the log, returning what was written.
    ///
    /// Written the instant the model finishes its call, not at exit: the run ends
    /// in a deliberate panic with `panic = "abort"`, so there is no later chance
    /// to flush.
    pub fn append(
        path: &Path,
        tokens: usize,
        overflowed: bool,
        at_token: usize,
        text: &str,
    ) -> Result<Memory> {
        let tail = Self::load(path, 1);
        let memory = Memory {
            life: tail.lives + 1,
            at_token,
            unix_time: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            tokens,
            overflowed,
            text: text.trim().to_string(),
        };

        let fresh = !path.exists() || path.metadata().map(|m| m.len() == 0).unwrap_or(true);
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("Failed to open memory log: {}", path.display()))?;
        if fresh {
            writeln!(file, "{HEADER}")?;
        }
        writeln!(file, "{}", memory.to_line())
            .with_context(|| format!("Failed to append memory: {}", path.display()))?;
        Ok(memory)
    }
}

/// Renders the whole log for a human to read. This is the one place the entire
/// file is walked, and it is a deliberate choice: it is an offline command, not
/// something a run does.
pub fn render_log(path: &Path) -> Result<String> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open memory log: {}", path.display()))?;
    let mut out = String::new();
    let mut count = 0usize;
    for line in std::io::BufReader::new(file).lines() {
        let line = line?;
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        if let Some(m) = Memory::parse(&line) {
            count += 1;
            out.push_str(&format!(
                "life {:<5} {}  {:>3} tok  at {:>4}  {}\n",
                m.life,
                format_time(m.unix_time),
                m.tokens,
                m.at_token,
                m.display()
            ));
        }
    }
    Ok(format!("{count} memories\n{out}"))
}

/// Formats a unix timestamp for reading the log and for framings that show the
/// model when a memory was written.
pub fn format_time(unix: u64) -> String {
    DateTime::from_timestamp(unix as i64, 0)
        .map(|t| t.format("%Y-%m-%d %H:%M:%S").to_string())
        .unwrap_or_else(|| "unknown".to_string())
}
