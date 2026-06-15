//! Golden tests: real prompts must rank the right skill first.
//!
//! Runs against the self-contained fixture skills under `tests/fixtures/skills/`.
//! Run with `--no-default-features` to use the offline bag-of-words embedder, so
//! it's network-free and deterministic. The default `fastembed` build exercises
//! the real bge embedder (model downloads on first run); the same prompts should
//! hold (and gain synonym tolerance) — add semantic-only prompts there later.

use ski::config::Config;
use ski::embed::{self, EmbedKind};
use ski::{index, rank, skill};
use std::path::PathBuf;

/// Bundled fixture skills, so the test depends on nothing outside this repo.
fn fixtures_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn top_skill(prompt: &str) -> String {
    let cfg = Config {
        roots: vec![fixtures_root()],
        ..Default::default()
    };
    let skills = skill::discover(&cfg.roots).expect("discover");
    assert!(
        !skills.is_empty(),
        "discovered no skills under {:?}",
        cfg.roots
    );
    let embedder = embed::build(&cfg.model).expect("embedder");
    let idx = index::build(&skills, embedder.as_ref(), None).expect("index");
    let query = embedder
        .embed(&[prompt.to_string()], EmbedKind::Query)
        .expect("embed")
        .remove(0);
    let hits = rank::rank_all(&query, prompt, &idx, &cfg);
    hits.first().expect("at least one hit").name.clone()
}

#[test]
fn discovers_fixture_skills() {
    let skills = skill::discover(&[fixtures_root()]).expect("discover");
    let names: Vec<&str> = skills.iter().map(|s| s.name.as_str()).collect();
    for expected in ["git-attribution", "uv-setup", "react-ts-setup", "handoff"] {
        assert!(
            names.contains(&expected),
            "missing skill {expected}; found {names:?}"
        );
    }
}

#[test]
fn golden_prompts_map_to_expected_skill() {
    let cases = [
        (
            "how should this git commit credit the AI assistant",
            "git-attribution",
        ),
        ("bootstrap a new python project with uv", "uv-setup"),
        (
            "scaffold a new react and typescript web app with vite",
            "react-ts-setup",
        ),
        ("write a handoff summary for the next agent", "handoff"),
        (
            "lemonade server not responding on its port",
            "debug-lemonade",
        ),
        (
            "add an ansible role to install a cli tool",
            "add-ansible-role",
        ),
    ];
    for (prompt, want) in cases {
        let got = top_skill(prompt);
        assert_eq!(got, want, "prompt {prompt:?} -> got {got:?}, want {want:?}");
    }
}
