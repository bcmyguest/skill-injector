//! Phrase-channel integration: a distinctive multi-word trigger phrase embedded
//! in a skill's description must (a) contribute a positive `phrase` score when the
//! user types it, lifting the right skill, and (b) contribute nothing on unrelated
//! prompts, so it cannot manufacture false positives.
//!
//! Uses the deterministic bag-of-words embedder directly, so the test is
//! network-free and feature-independent — the phrase channel is lexical and lives
//! in the ranker, not the embedder, so this isolates exactly the new signal.

use ski::config::Config;
use ski::embed::bow::BowEmbedder;
use ski::embed::{EmbedKind, Embedder};
use ski::{index, rank, skill};
use std::fs;
use std::path::PathBuf;

fn write_skill(root: &std::path::Path, name: &str, desc: &str) {
    let dir = root.join(name);
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        dir.join("SKILL.md"),
        format!("---\nname: {name}\ndescription: {desc}\n---\nbody\n"),
    )
    .unwrap();
}

// A per-test directory. The two tests in this file share one process (one PID),
// so a PID-only name let them collide on the same path — one test's
// `remove_dir_all` could wipe the other's tree mid-run under the parallel
// runner. The `label` keeps the two tests apart and the nanos suffix keeps
// successive runs apart (mirrors the helper idiom in `src/skill.rs`).
fn temp_root(label: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let d = std::env::temp_dir().join(format!(
        "ski-phrase-it-{}-{label}-{nanos}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

/// Score every skill for `prompt` with the given phrase boost (0.0 disables the
/// channel), returning the hits sorted by the ranker.
fn rank_with(root: &std::path::Path, prompt: &str, phrase_boost: f32) -> Vec<rank::Hit> {
    let cfg = Config {
        roots: vec![root.to_path_buf()],
        phrase_boost,
        ..Default::default()
    };
    let skills = skill::discover(&cfg.roots).expect("discover");
    let embedder = BowEmbedder::new();
    let idx = index::build(&skills, &embedder, None).expect("index");
    let query = embedder
        .embed(&[prompt.to_string()], EmbedKind::Query)
        .unwrap()
        .remove(0);
    rank::rank_all(&query, prompt, &idx, &cfg)
}

fn rank(root: &std::path::Path, prompt: &str) -> Vec<rank::Hit> {
    rank_with(root, prompt, Config::default().phrase_boost)
}

#[test]
fn trigger_phrase_lifts_its_skill_and_stays_silent_elsewhere() {
    let root = temp_root("trigger");
    // A skill whose *only* distinctive trigger is a quoted multi-word phrase.
    write_skill(
        &root,
        "web-search",
        "Search the public internet for information. Use when the user says \"find that page online\" or wants current facts.",
    );
    // A decoy that shares no trigger phrase.
    write_skill(
        &root,
        "git-attribution",
        "Credit AI assistance in git commits following the kernel policy.",
    );

    // Exact-trigger prompt: the phrase channel must fire for web-search.
    let hits = rank(&root, "can you find that page online for me");
    let ws = hits
        .iter()
        .find(|h| h.id == "web-search")
        .expect("web-search ranked");
    assert!(
        ws.phrase > 0.0,
        "phrase channel did not fire on the exact trigger: {ws:?}"
    );
    assert_eq!(hits[0].id, "web-search", "trigger prompt: {hits:?}");

    // Unrelated prompt sharing a single phrase token ("online"): must NOT fire.
    let hits = rank(&root, "take my service online and deploy it");
    let ws = hits.iter().find(|h| h.id == "web-search").unwrap();
    assert_eq!(
        ws.phrase, 0.0,
        "phrase channel false-fired on partial overlap: {ws:?}"
    );

    let _ = fs::remove_dir_all(&root);
}

/// A/B: turning the phrase channel on must strictly raise the right skill's score
/// on an exact-trigger prompt (true-positive lift), while leaving every skill's
/// phrase contribution at zero on a prompt that only partially overlaps a trigger
/// (no false-positive lift). Holds the embedder fixed, so the delta is purely the
/// new channel.
#[test]
fn phrase_channel_lifts_true_positive_without_lifting_false_positive() {
    let root = temp_root("ab");
    write_skill(
        &root,
        "accessibility",
        "Make web UIs usable by everyone. Use when the user asks to \"add screen reader support\" or audit contrast.",
    );
    write_skill(
        &root,
        "pdf-tools",
        "Read and assemble PDF files. Use to \"merge several pdf documents\" or extract pages.",
    );

    // Exact trigger -> score with the channel must exceed score without it.
    let trigger = "please add screen reader support to my signup form";
    let on = rank_with(&root, trigger, 0.20);
    let off = rank_with(&root, trigger, 0.0);
    let on_a = on.iter().find(|h| h.id == "accessibility").unwrap();
    let off_a = off.iter().find(|h| h.id == "accessibility").unwrap();
    assert!(
        on_a.score > off_a.score,
        "phrase channel did not lift the true positive: on={on_a:?} off={off_a:?}"
    );
    assert_eq!(on[0].id, "accessibility");

    // Partial-overlap negative: shares "support" and "pdf"/"screen" with neither
    // full trigger -> no skill may gain a phrase contribution.
    let neg = "my laptop screen flickers and tech support was no help";
    for h in rank_with(&root, neg, 0.20) {
        assert_eq!(
            h.phrase, 0.0,
            "phrase fired on a partial-overlap negative: {h:?}"
        );
    }

    let _ = fs::remove_dir_all(&root);
}
