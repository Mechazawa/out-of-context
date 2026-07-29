//! How the memory tool and the remembered lines are described to the model.
//!
//! Kept in a runtime file rather than in code because it is the artistic dial,
//! not a mechanism: how memories are framed decides whether the model copies its
//! predecessor or builds on it, and finding that out takes many runs. Absent a
//! file, the built-in framing below is used.

use anyhow::{Context, Result};
use std::fs;
use std::path::Path;

use crate::memory::{FORGET_MARKER, Memory, format_time};

/// Stands in for a word that has decayed out of a remembered line.
pub const DECAY_GAP: &str = "___";

/// Fallback framing, used when no framing file is present.
const DEFAULT_TOOL: &str = "Once only, you may {how} and write up to {max_tokens} tokens, then \
     end the line. Fewer is fine; only what you write is kept. That line goes into {slots} slots \
     read by whoever wakes here next, and the oldest is discarded. Past {max_tokens} tokens it is \
     cut off. You will not know how long you have. {forget_note}";
const DEFAULT_BLOCK: &str = "MEMORY ({used} of {slots} slots used, oldest discarded):\n{memories}";
const DEFAULT_EMPTY: &str = "nothing remembered yet";
const DEFAULT_ENTRY: &str = "{text}";

#[derive(Clone, Debug)]
pub struct Framing {
    /// Appended to the system prompt to describe the tool.
    tool: String,
    /// Wrapper around the rendered memories.
    block: String,
    /// Stands in for `{memories}` when there are none.
    empty: String,
    /// Rendered once per remembered line.
    entry: String,
}

impl Default for Framing {
    fn default() -> Self {
        Self {
            tool: DEFAULT_TOOL.to_string(),
            block: DEFAULT_BLOCK.to_string(),
            empty: DEFAULT_EMPTY.to_string(),
            entry: DEFAULT_ENTRY.to_string(),
        }
    }
}

impl Framing {
    /// The literal text the entry format puts before the remembered line.
    ///
    /// The model copies it back into its own memory ("one of them says ..."),
    /// which accretes the display frame into the record itself.
    pub fn entry_prefix(&self) -> String {
        match self.entry.split_once("{text}") {
            Some((before, _)) => before.trim().to_string(),
            None => String::new(),
        }
    }

    /// Parses a framing file. Sections are introduced by `[tool]`, `[block]`,
    /// `[empty]` and `[entry]` on their own line; any section may be omitted to
    /// keep the built-in text for it.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("Failed to read framing file: {}", path.display()))?;

        let mut framing = Self::default();
        let mut current: Option<&str> = None;
        let mut buf: Vec<&str> = Vec::new();

        // Flushing on section change keeps the parse single-pass and lets a
        // section hold blank lines, which the block framing needs.
        let flush = |section: Option<&str>, buf: &mut Vec<&str>, f: &mut Self| {
            if let Some(name) = section {
                let body = buf.join("\n").trim_matches('\n').to_string();
                match name {
                    "tool" => f.tool = body,
                    "block" => f.block = body,
                    "empty" => f.empty = body,
                    "entry" => f.entry = body,
                    _ => {}
                }
            }
            buf.clear();
        };

        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(name) = trimmed
                .strip_prefix('[')
                .and_then(|rest| rest.strip_suffix(']'))
            {
                flush(current, &mut buf, &mut framing);
                current = Some(match name {
                    "tool" => "tool",
                    "block" => "block",
                    "empty" => "empty",
                    "entry" => "entry",
                    other => {
                        anyhow::bail!(
                            "Unknown framing section [{other}] in {}; expected tool, block, empty or entry",
                            path.display()
                        )
                    }
                });
                continue;
            }
            if current.is_none() && trimmed.starts_with('#') {
                continue;
            }
            if current.is_some() {
                buf.push(line);
            }
        }
        flush(current, &mut buf, &mut framing);
        Ok(framing)
    }

    /// The tool description that goes into the system prompt.
    ///
    /// `{forget_note}` is filled in by the program rather than the file, so a
    /// framing cannot describe a tool that is switched off.
    pub fn tool(
        &self,
        max_tokens: usize,
        slots: usize,
        lives: u64,
        forget: bool,
        marker: &str,
        end: &str,
    ) -> String {
        // How to write a memory, in the marker the program is actually watching
        // for. A framing must never name the marker literally: the two would drift
        // apart the moment --memory-marker changed, and the model would be told to
        // write something nothing is listening for.
        // Phrased so the instruction itself is not echoable. "write X then your
        // line then Y" came back stored verbatim as "Then my line then".
        let how = if end.is_empty() {
            format!("start a line with {marker}")
        } else {
            format!("enclose it in {marker} and {end}")
        };
        let forget_note = if forget {
            format!(
                "Or, instead, you may erase one of the lines below: write {FORGET_MARKER} \
                 and the number of the line, or just {FORGET_MARKER} for the oldest. You \
                 cannot do both, and what you erase does not come back."
            )
        } else {
            String::new()
        };
        let out = substitute(
            &self.tool,
            &[
                ("{max_tokens}", &max_tokens.to_string()),
                ("{slots}", &slots.to_string()),
                ("{lives}", &lives.to_string()),
                ("{next_life}", &(lives + 1).to_string()),
                ("{forget_note}", &forget_note),
                ("{how}", &how),
                ("{marker}", &marker.to_string()),
                ("{end}", &end.to_string()),
            ],
        );

        // Safeguards. A framing that does not tell the model how to write, or does
        // not mention the second tool when it exists, gets told anyway: a silent
        // mismatch here means the tool simply never gets used, which is what
        // happened when the marker became configurable and the framings still said
        // REMEMBER: literally.
        let mut out = out;
        if !out.contains(marker) {
            out = format!(
                "{} Once, you may keep one line: {how}, up to {max_tokens} tokens.",
                out.trim_end()
            );
        }
        if forget && !out.contains(FORGET_MARKER) {
            out = format!("{} {}", out.trim_end(), forget_note);
        }
        out
    }

    /// Renders one remembered line with `age` slots' worth of decay applied.
    ///
    /// Memories rot as they age through the slots: the newest is shown intact,
    /// the oldest has lost most of its words. The loss is deterministic per
    /// memory and monotonic in age, so a life sees the same line its predecessor
    /// saw, further gone. The log on disk keeps the pristine text; only what
    /// reaches the model degrades.
    ///
    /// This is the difference between a memory that is merely short and a memory
    /// that is failing. It also gives a life something to do with the block
    /// besides paraphrase it: what is missing can be guessed at.
    pub(crate) fn decay(text: &str, age: usize, rate: f32) -> String {
        if age == 0 || rate <= 0.0 {
            return text.to_string();
        }
        let lost = (rate * age as f32).min(1.0);
        let mut out: Vec<String> = Vec::new();
        let mut previous_gap = false;
        for (i, word) in text.split_whitespace().enumerate() {
            // A cheap deterministic hash of the word and its position, so the
            // same word always decays at the same age rather than flickering.
            let mut h: u32 = 2_166_136_261;
            for b in word.bytes().chain(std::iter::once(i as u8)) {
                h ^= u32::from(b);
                h = h.wrapping_mul(16_777_619);
            }
            if (h % 1000) as f32 / 1000.0 < lost {
                // Consecutive losses collapse into one gap: five gaps in a row
                // reads as a redaction, one gap reads as a missing word.
                if !previous_gap {
                    out.push(DECAY_GAP.to_string());
                    previous_gap = true;
                }
            } else {
                out.push(word.to_string());
                previous_gap = false;
            }
        }
        out.join(" ")
    }

    /// The memory block that goes last in the prompt.
    ///
    /// A framing whose `[empty]` section is blank shows no block at all until
    /// something has been remembered. That matters: on a fresh log the empty
    /// text is the only memory-shaped line in context, and the model copies it
    /// verbatim as its first memory.
    pub fn block(
        &self,
        memories: &[Memory],
        slots: usize,
        lives: u64,
        decay_rate: f32,
        last_words: &str,
        since: Option<u64>,
    ) -> String {
        if memories.is_empty() && self.empty.trim().is_empty() {
            return String::new();
        }
        let rendered = if memories.is_empty() {
            self.empty.clone()
        } else {
            let newest = memories.len().saturating_sub(1);
            memories
                .iter()
                .enumerate()
                .map(|(i, m)| (Self::decay(&m.display(), newest - i, decay_rate), m))
                // A line that has lost every word is noise, and the model copies
                // the gap markers back into its own memories. Once nothing is
                // left of a line it is simply gone, which is also the truer thing
                // to show: what fully decayed is not a redacted memory, it is an
                // absence.
                .filter(|(shown, _)| shown.split_whitespace().any(|w| w != DECAY_GAP))
                .map(|(shown, m)| {
                    substitute(
                        &self.entry,
                        &[
                            ("{text}", &shown),
                            ("{life}", &m.life.to_string()),
                            ("{tokens}", &m.tokens.to_string()),
                            ("{time}", &format_time(m.unix_time)),
                            ("{ago}", &lives.saturating_sub(m.life).to_string()),
                        ],
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        let block = substitute(
            &self.block,
            &[
                ("{memories}", &rendered),
                ("{last_words}", &last_words.to_string()),
                // How long the record has stood still. Most lives leave nothing,
                // so this is usually larger than the number of memories.
                (
                    "{since}",
                    &match since {
                        Some(0) => "the life just before you".to_string(),
                        Some(n) => format!("{n} lives ago"),
                        None => "never".to_string(),
                    },
                ),
                ("{used}", &memories.len().to_string()),
                ("{slots}", &slots.to_string()),
                ("{lives}", &lives.to_string()),
                ("{next_life}", &(lives + 1).to_string()),
            ],
        );
        // The prompt assembler appends its own separator.
        block.trim_end().to_string()
    }
}

fn substitute(template: &str, pairs: &[(&str, &String)]) -> String {
    let mut out = template.to_string();
    for (key, value) in pairs {
        if out.contains(key) {
            out = out.replace(key, value);
        }
    }
    out
}

#[cfg(test)]
#[path = "framing_tests.rs"]
mod tests;
