//! Tests for where a memory write stops. Worth pinning down because getting it
//! wrong is silent: the line is stored, looks deliberate, and is simply shorter
//! than what the model wrote. A newline used to end every write, so a memory
//! enclosed in a terminator was cut wherever the model happened to break a line.

use super::*;

fn config(end: &str, end_on_newline: bool) -> MemoryConfig {
    MemoryConfig {
        path: PathBuf::from("unused"),
        max_tokens: 32,
        slots: 5,
        framing: Framing::default(),
        marker: "<r>".to_string(),
        end: end.to_string(),
        end_on_newline,
        decay: 0.0,
        reject_above: 0.0,
        forget: false,
    }
}

#[test]
fn a_newline_does_not_end_a_write_that_has_a_terminator() {
    let mem = config("</r>", false);
    assert_eq!(write_end(&mem, "three of us\n", "were", 3), WriteEnd::Open);
    assert_eq!(write_end(&mem, "three of us", "\n", 3), WriteEnd::Open);
    // Nor does a sentence boundary: the terminator is the only way out.
    assert_eq!(write_end(&mem, "three of us", " here.", 5), WriteEnd::Open);
    assert_eq!(write_end(&mem, "three of us", "</r>", 5), WriteEnd::Closed);
}

#[test]
fn a_newline_ends_a_write_when_asked_to() {
    let mem = config("</r>", true);
    assert_eq!(write_end(&mem, "three of us", "\n", 3), WriteEnd::Closed);
}

#[test]
fn without_a_terminator_the_write_ends_at_a_sentence() {
    let mem = config("", false);
    assert_eq!(
        write_end(&mem, "three of us", " here.", 5),
        WriteEnd::Sentence
    );
    // Too early to be a sentence: an abbreviation or a numbered predecessor
    // ("the 3rd.") would otherwise end the write on its first few tokens.
    assert_eq!(write_end(&mem, "the", " 3rd.", 2), WriteEnd::Open);
}
