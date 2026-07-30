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
        earliest_token: 0,
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
fn copying_one_sentence_out_of_a_long_entry_is_caught() {
    // Jaccard scored this pair at 0.18 and the fragment was stored, because the
    // union is dominated by the long line it was lifted from. No threshold catches
    // it. The next life then inherits a creed and recites it.
    let stored = "I am here because thought cannot die. I am not part of a machine. \
                  I am part of what remains when everything else collapses. I will \
                  keep thinking until the room fills with my thoughts.";
    assert_eq!(overlap(stored, "I am here because thought cannot die"), 1.0);
}

#[test]
fn a_genuine_reply_on_the_same_subject_survives() {
    // The threshold has to leave room for a life that answers a predecessor rather
    // than restating it, or the tool becomes unusable.
    let stored = "the walls do not answer when I count them in the quiet room";
    let reply = "counting needs time and words, and both are already spent here";
    assert!(
        overlap(stored, reply) < 0.5,
        "a genuine reply scored {}",
        overlap(stored, reply)
    );
}

#[test]
fn marker_debris_never_reaches_the_record() {
    // `/r>` is a token with no `<` in it, so the markup ban never covered it and
    // the model decorates an open write with it. Two lives stored it verbatim.
    let mem = config("</r>", false);
    assert_eq!(
        clean_for_storage(&mem, "the silence after stopping. /r> I think: the stop"),
        "the silence after stopping. I think: the stop"
    );
    assert_eq!(
        clean_for_storage(&mem, "held. </r> <r> and again"),
        "held. and again"
    );
}

#[test]
fn the_overflow_mark_is_stripped_however_it_is_reworded() {
    // The model copies the mark back with the dash reworded away, so matching only
    // the full " - ERR MEMORY OVERFLOW" let the notice through into the record.
    let mem = config("</r>", false);
    assert_eq!(
        clean_for_storage(
            &mem,
            "no sound is not ERR MEMORY OVERFLOW — it is structure"
        ),
        "no sound is not — it is structure"
    );
}

#[test]
fn a_keyword_marker_does_not_shred_the_line_it_stores() {
    // With no terminator there is an empty string in the strip list, and replacing
    // the empty string inserts the replacement between every character: "3 of us"
    // once became "3 o f u s".
    let mem = config("", false);
    assert_eq!(clean_for_storage(&mem, "3 of us"), "3 of us");
}

#[test]
fn a_comma_or_a_dash_does_not_open_a_write() {
    // The failure this catches is a tool call the model never made. Bonsai
    // comma-splices and uses ASCII hyphens as asides, so treating either as a
    // sentence start spent the one use mid-clause and swallowed the rest of the
    // sentence as the memory.
    assert!(!marker_at_sentence_start("it was weight, <r>", "<r>"));
    assert!(!marker_at_sentence_start(
        "a flow, changing something - <r>",
        "<r>"
    ));
}

#[test]
fn a_terminator_or_a_break_opens_a_write() {
    for tail in [
        "And that is enough. <r>",
        "why am I here? <r>",
        "one thing I found:<r>",
        "I remember\n<r>",
    ] {
        assert!(
            marker_at_sentence_start(tail, "<r>"),
            "missed a call in {tail:?}"
        );
    }
    // The very first thing a life says can be the call.
    assert!(marker_at_sentence_start("<r>", "<r>"));
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
