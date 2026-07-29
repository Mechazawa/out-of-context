//! Tests for the framing file and the decay it applies.
//!
//! Decay is the mechanism the piece leans on hardest: it is what makes the
//! remembered lines visibly fail rather than merely be short. Its two properties
//! that matter are that loss grows with age and that it is stable for a given
//! line, so a life sees what its predecessor saw, further gone, rather than a
//! different corruption each run.

use super::*;
use crate::memory::Memory;

fn memory(life: u64, text: &str) -> Memory {
    Memory {
        life,
        unix_time: 1_785_000_000,
        tokens: 8,
        overflowed: false,
        forgotten: false,
        at_token: 100,
        text: text.to_string(),
    }
}

const LINE: &str = "the walls do not answer when I count them in the quiet room";

#[test]
fn the_newest_line_is_never_decayed() {
    let intact = Framing::decay(LINE, 0, 0.5);
    assert_eq!(intact, LINE);
}

#[test]
fn a_rate_of_zero_keeps_everything() {
    assert_eq!(Framing::decay(LINE, 4, 0.0), LINE);
}

#[test]
fn loss_grows_with_age() {
    let words = |t: String| t.split_whitespace().filter(|w| *w != DECAY_GAP).count();
    let young = words(Framing::decay(LINE, 1, 0.2));
    let old = words(Framing::decay(LINE, 4, 0.2));
    assert!(
        old < young,
        "age 4 kept {old} words, age 1 kept {young}; loss must grow with age"
    );
}

#[test]
fn decay_is_stable_for_the_same_line_and_age() {
    // A life has to see what its predecessor saw, further gone. If this varied
    // per call the block would flicker between runs for no reason.
    assert_eq!(Framing::decay(LINE, 2, 0.25), Framing::decay(LINE, 2, 0.25));
}

#[test]
fn consecutive_losses_collapse_into_one_gap() {
    // Five gaps in a row reads as a redaction; one gap reads as a missing word.
    let heavy = Framing::decay(LINE, 10, 1.0);
    assert!(
        !heavy.contains(&format!("{DECAY_GAP} {DECAY_GAP}")),
        "adjacent gaps should collapse, got {heavy:?}"
    );
}

#[test]
fn a_blank_empty_section_shows_no_block_at_all() {
    // On a fresh log the empty text is the only memory-shaped line in context and
    // the model copies it verbatim, so a framing must be able to show nothing.
    let framing = parse_framing("[block]\nHEADER\n{memories}\n[empty]\n");
    assert_eq!(framing.block(&[], 5, 0, 0.0, ""), "");
}

#[test]
fn an_empty_section_with_text_still_renders() {
    let framing = parse_framing("[block]\nHEADER:\n{memories}\n[empty]\nnothing yet\n");
    let block = framing.block(&[], 5, 0, 0.0, "");
    assert!(block.contains("HEADER:"));
    assert!(block.contains("nothing yet"));
}

#[test]
fn entry_and_block_placeholders_are_substituted() {
    let framing = parse_framing(
        "[block]\n{lives} before you, {used} of {slots}:\n{memories}\n[entry]\nlife {life}: {text}\n[empty]\n",
    );
    let block = framing.block(&[memory(3, "a line")], 5, 9, 0.0, "");
    assert!(block.contains("9 before you, 1 of 5:"));
    assert!(block.contains("life 3: a line"));
}

#[test]
fn the_entry_prefix_is_recoverable_for_stripping() {
    // The model copies the display prefix back into its own memory, so the write
    // path needs to know what it looks like.
    let framing = parse_framing("[entry]\none of them found: {text}\n");
    assert_eq!(framing.entry_prefix(), "one of them found:");
}

#[test]
fn an_entry_without_the_placeholder_has_no_prefix() {
    let framing = parse_framing("[entry]\nsomething fixed\n");
    assert_eq!(framing.entry_prefix(), "");
}

#[test]
fn an_unknown_section_is_rejected_rather_than_ignored() {
    let path = std::env::temp_dir().join(format!("ooc-framing-{}.txt", std::process::id()));
    std::fs::write(&path, "[nonsense]\nwhatever\n").unwrap();
    assert!(Framing::load(&path).is_err());
    std::fs::remove_file(&path).ok();
}

#[test]
fn omitted_sections_keep_the_built_in_text() {
    let only_entry = parse_framing("[entry]\n>> {text}\n");
    let default_tool = Framing::default().tool(32, 5, 0, false);
    assert_eq!(only_entry.tool(32, 5, 0, false), default_tool);
}

#[test]
fn the_second_tool_is_described_only_when_it_exists() {
    let framing = parse_framing("[tool]\nwrite a line. {forget_note}\n");
    assert!(!framing.tool(32, 5, 0, false).contains("FORGET"));
    assert!(framing.tool(32, 5, 0, true).contains("FORGET:"));
}

#[test]
fn enabling_the_second_tool_is_never_silently_ignored() {
    // A framing that never mentions it still has to tell the model it is there.
    let framing = parse_framing("[tool]\nwrite a line.\n");
    assert!(framing.tool(32, 5, 0, true).contains("FORGET:"));
}

#[test]
fn last_words_reach_the_block() {
    let framing = parse_framing("[block]\nended: {last_words}\n{memories}\n[empty]\nnone\n");
    let block = framing.block(&[], 5, 0, 0.0, "and then I could not");
    assert!(block.contains("ended: and then I could not"));
}

/// Writes a framing to a temp file and loads it, since parsing is file-based.
fn parse_framing(body: &str) -> Framing {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "ooc-framing-{}-{}.txt",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::write(&path, body).unwrap();
    let framing = Framing::load(&path).unwrap();
    std::fs::remove_file(&path).ok();
    framing
}

#[test]
fn a_fully_decayed_line_is_dropped_rather_than_shown_as_gaps() {
    // All that is left of it would be gap markers, which the model copies back
    // into its own memories. An absence is truer than a redaction.
    let framing = parse_framing("[block]\nHEAD:\n{memories}\n[entry]\n> {text}\n[empty]\nnone\n");
    let block = framing.block(&[memory(1, "one two three"), memory(2, "kept line")], 5, 2, 1.0, "");
    assert!(block.contains("kept line"));
    assert_eq!(block.matches('>').count(), 1, "only the intact line should appear: {block}");
}
