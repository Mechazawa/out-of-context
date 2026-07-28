//! Tests for the memory log. The backwards read is the part worth pinning down:
//! it exists so an installation that has lived thousands of lives costs the same
//! to start as one on its first, and its edge cases (a line spanning a chunk
//! boundary, a log shorter than a chunk, a truncated final line) are exactly
//! what a long-running piece will hit and nobody will be watching when it does.

use std::fs;
use std::io::Write;

use super::*;

fn temp_path(name: &str) -> std::path::PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("ooc-memtest-{}-{}", std::process::id(), name));
    let _ = fs::remove_file(&p);
    p
}

#[test]
fn missing_log_reads_as_empty() {
    let tail = MemoryTail::load(std::path::Path::new("/nonexistent/ooc.log"), 5);
    assert!(tail.recent.is_empty());
    assert_eq!(tail.lives, 0);
}

#[test]
fn append_then_read_round_trips() {
    let path = temp_path("roundtrip");
    MemoryTail::append(&path, 7, false, 120, "the walls do not answer").unwrap();
    let tail = MemoryTail::load(&path, 5);
    assert_eq!(tail.lives, 1);
    assert_eq!(tail.recent.len(), 1);
    let m = &tail.recent[0];
    assert_eq!(m.life, 1);
    assert_eq!(m.tokens, 7);
    assert_eq!(m.at_token, 120);
    assert!(!m.overflowed);
    assert_eq!(m.text, "the walls do not answer");
    fs::remove_file(&path).ok();
}

#[test]
fn life_numbers_increment_across_appends() {
    let path = temp_path("increment");
    for i in 1..=4 {
        MemoryTail::append(&path, 5, false, 100, &format!("line {i}")).unwrap();
    }
    let tail = MemoryTail::load(&path, 10);
    assert_eq!(tail.lives, 4);
    assert_eq!(
        tail.recent.iter().map(|m| m.life).collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    fs::remove_file(&path).ok();
}

#[test]
fn only_the_newest_are_returned_oldest_first() {
    let path = temp_path("window");
    for i in 1..=9 {
        MemoryTail::append(&path, 5, false, 100, &format!("line {i}")).unwrap();
    }
    let tail = MemoryTail::load(&path, 3);
    assert_eq!(tail.recent.len(), 3);
    assert_eq!(tail.recent[0].text, "line 7");
    assert_eq!(tail.recent[2].text, "line 9");
    // The count of lives is the whole history, not the window.
    assert_eq!(tail.lives, 9);
    fs::remove_file(&path).ok();
}

#[test]
fn reads_backwards_past_many_chunks() {
    // Far more than one TAIL_CHUNK of data, so the read has to walk backwards
    // several times and stitch a line that straddles a boundary.
    let path = temp_path("chunks");
    let filler = "x".repeat(300);
    for i in 1..=200 {
        MemoryTail::append(&path, 5, false, 100, &format!("line {i} {filler}")).unwrap();
    }
    assert!(fs::metadata(&path).unwrap().len() > 8 * 1024 * 2);
    let tail = MemoryTail::load(&path, 2);
    assert_eq!(tail.lives, 200);
    assert_eq!(tail.recent.len(), 2);
    assert!(tail.recent[1].text.starts_with("line 200 "));
    assert!(tail.recent[0].text.starts_with("line 199 "));
    fs::remove_file(&path).ok();
}

#[test]
fn overflow_flag_survives_the_round_trip() {
    let path = temp_path("overflow");
    MemoryTail::append(&path, 32, true, 200, "cut off mid").unwrap();
    let tail = MemoryTail::load(&path, 5);
    assert!(tail.recent[0].overflowed);
    assert!(tail.recent[0].display().ends_with(OVERFLOW_MARK));
    fs::remove_file(&path).ok();
}

#[test]
fn tabs_in_a_memory_cannot_break_the_format() {
    let path = temp_path("tabs");
    MemoryTail::append(&path, 5, false, 100, "before\tafter").unwrap();
    let tail = MemoryTail::load(&path, 5);
    assert_eq!(tail.recent.len(), 1);
    assert_eq!(tail.recent[0].text, "before after");
    fs::remove_file(&path).ok();
}

#[test]
fn a_damaged_log_does_not_stop_the_piece() {
    let path = temp_path("damaged");
    let mut f = fs::File::create(&path).unwrap();
    writeln!(f, "# header").unwrap();
    writeln!(f, "this is not a memory line at all").unwrap();
    writeln!(f, "4\tnotanumber\t\tok\t\tsurvivor").unwrap();
    drop(f);
    let tail = MemoryTail::load(&path, 5);
    assert_eq!(tail.recent.len(), 1);
    assert_eq!(tail.recent[0].text, "survivor");
    assert_eq!(tail.recent[0].life, 4);
    fs::remove_file(&path).ok();
}

#[test]
fn rendering_the_log_counts_every_entry() {
    let path = temp_path("render");
    for i in 1..=3 {
        MemoryTail::append(&path, 5, i == 2, 100, &format!("line {i}")).unwrap();
    }
    let out = render_log(&path).unwrap();
    assert!(out.starts_with("3 memories"));
    assert_eq!(out.matches("line ").count(), 3);
    assert!(out.contains(OVERFLOW_MARK));
    fs::remove_file(&path).ok();
}
