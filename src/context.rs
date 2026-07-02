//! Conversational-context query enrichment (the query side of retrieval).
//!
//! A vague follow-up prompt ("now do the other one", "fix that") carries little
//! signal on its own, so the bi-encoder retrieves poorly and the cross-encoder
//! has nothing to disambiguate against. The turns *before* it usually do carry
//! the intent. This module turns a session's recent-prompt window into two
//! enrichment signals, both gated on how vague the current prompt is
//! ([`crate::rank::context_weight`]):
//!
//! - a **context vector** ([`vector`]) blended into stage-1 scoring
//!   ([`crate::rank::rank_all_ctx`]), and
//! - an enriched **reranker query** ([`rerank_query`]) — the cross-encoder reads
//!   text, not vectors, so the recent window is prepended to the prompt.
//!
//! Both are inert unless the feature is enabled (`context_depth > 0` and
//! `context_weight > 0.0`), so the default path pays nothing.

use crate::config::Config;
use crate::embed::{EmbedKind, Embedder};
use crate::index::Index;
use crate::text::{match_tokens, norm_token, tokenize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// Skill ids implied by a file extension, for the file-type context channel. Only
/// extensions whose document *kind* is an unambiguous 1:1 with a skill are mapped —
/// a `.xlsx`/`.ods`/`.numbers` is a spreadsheet task, a `.key`/`.odp` is a deck — so
/// the boost stays high-precision and routes by intent even for formats a skill's
/// own tooling converts rather than opens natively. Generic code extensions
/// (`.rs`, `.py`, ...) map to no single skill and are deliberately absent. Images
/// (`.png`/`.jpg`/`.gif`) and notebooks (`.ipynb`) are excluded for the same
/// precision reason: no installed skill is their unambiguous identity (image skills
/// are intent-specific, and there is no notebook skill), so mapping them would buy
/// recall with false-injects.
fn ext_skill(ext: &str) -> Option<&'static str> {
    match ext {
        "pdf" => Some("pdf"),
        // Spreadsheet identity: the formats the xlsx skill reads (csv/tsv) plus the
        // OpenDocument/iWork/legacy spreadsheet equivalents.
        "xlsx" | "xls" | "xlsm" | "csv" | "tsv" | "ods" | "numbers" => Some("xlsx"),
        // Word-processor identity.
        "docx" | "doc" | "rtf" | "odt" | "pages" => Some("docx"),
        // Presentation identity.
        "pptx" | "ppt" | "odp" | "key" => Some("pptx"),
        _ => None,
    }
}

/// Iterate the filename-shaped tokens of `text`, yielding `(stem, extension)` for
/// each token carrying a `.<ext>` suffix. Shared by the file-type channel
/// ([`file_ids`]) and the code-file ecosystem scan ([`code_terms`]).
fn file_tokens(text: &str) -> impl Iterator<Item = (&str, String)> {
    text.split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '(' | ')' | '`' | ','))
        .filter_map(|tok| {
            // Strip trailing punctuation that commonly hugs a filename in prose.
            let tok = tok.trim_end_matches(['.', ':', ';', '!', '?']);
            let (stem, ext) = tok.rsplit_once('.')?;
            if stem.is_empty() {
                return None; // a bare ".pdf" with no name is not a real reference.
            }
            Some((stem, ext.to_ascii_lowercase()))
        })
}

/// Skill ids implied by file references in `text` (a prompt and/or recent-window
/// turns): scans whitespace-separated tokens for a trailing `.<ext>` and maps each
/// known extension through [`ext_skill`]. This is the *directly attributable*
/// context signal — a file attached or named **now** is unambiguous in a way a
/// vague prompt is not, so unlike the dense context vector it is not gated on
/// prompt vagueness. De-duplicated.
pub fn file_ids(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (_, ext) in file_tokens(text) {
        if let Some(id) = ext_skill(&ext) {
            out.insert(id.to_string());
        }
    }
    out
}

/// Known project-manifest filenames and the ecosystem terms each implies, for the
/// project-type context channel. Unlike the file channel's 1:1 skill-id map, these
/// are *terms* matched dynamically against whatever library the user actually has
/// installed ([`skills_for_terms`]) — a `uv.lock` surfaces *their* uv skill
/// whatever it is named, and an unmatched term simply maps to nothing. That is why
/// multi-skill ecosystems (python, the JS frameworks) can be listed here where the
/// old hardcoded-id map had to leave them out: every plausibly-matching skill gets
/// the (cosine-gated, deliberately recall-leaning) boost and the model arbitrates.
const MANIFEST_TERMS: &[(&str, &[&str])] = &[
    ("Cargo.toml", &["rust", "cargo"]),
    ("go.mod", &["go", "golang"]),
    ("uv.lock", &["uv", "python"]),
    ("pyproject.toml", &["python"]),
    ("requirements.txt", &["python", "pip"]),
    ("setup.py", &["python"]),
    ("Pipfile", &["python"]),
    ("package.json", &["javascript", "node", "npm"]),
    ("tsconfig.json", &["typescript"]),
    ("Gemfile", &["ruby"]),
    ("pom.xml", &["java", "maven"]),
    ("build.gradle", &["java", "gradle"]),
    ("build.gradle.kts", &["kotlin", "gradle"]),
    ("Dockerfile", &["docker"]),
    ("docker-compose.yml", &["docker"]),
    ("compose.yaml", &["docker"]),
    ("flake.nix", &["nix"]),
    ("CMakeLists.txt", &["cmake"]),
];

/// Ecosystem terms implied by a *code* file's extension, feeding the same ambient
/// project channel as the manifests. This covers the session working outside the
/// project root (cwd walk finds nothing) but naming `scripts/etl.py` in the
/// prompt, and attached code files. Kept to unambiguous language identities;
/// document formats stay with the higher-precision, ungated [`ext_skill`] channel.
fn ext_terms(ext: &str) -> Option<&'static [&'static str]> {
    Some(match ext {
        "py" => &["python"],
        "ipynb" => &["python", "jupyter", "notebook"],
        "rs" => &["rust"],
        "go" => &["go", "golang"],
        "ts" | "tsx" => &["typescript"],
        "js" | "jsx" | "mjs" => &["javascript", "node"],
        "rb" => &["ruby"],
        "java" => &["java"],
        "kt" => &["kotlin"],
        "tf" => &["terraform"],
        "sql" => &["sql"],
        "sh" | "bash" => &["shell", "bash"],
        _ => return None,
    })
}

/// How many directory levels to walk upward from `cwd` looking for a manifest. A
/// session's cwd is often a subdirectory of the project root where the manifest
/// lives, so we ascend a few levels — but cap it so a deeply-nested cwd cannot
/// stat its way to the filesystem root.
const PROJECT_WALK_LEVELS: usize = 6;

/// Append `term` if it isn't already present — an order-preserving de-dup, so a
/// term list keeps most-specific-first ordering (the order [`MANIFEST_TERMS`] /
/// [`ext_terms`] list them in) for [`skills_for_terms`]'s first-match-wins
/// evidence attribution: a uv.lock reports "a uv project", not "a python project".
fn push_term(out: &mut Vec<String>, term: &str) {
    if !out.iter().any(|t| t == term) {
        out.push(term.to_string());
    }
}

/// Ecosystem terms implied by the project manifest(s) found in `cwd` or any
/// ancestor directory (up to [`PROJECT_WALK_LEVELS`]). Performs cheap `exists()`
/// stats only; order-preserving and de-duplicated (most specific term first);
/// empty when `cwd` is empty or no known manifest is found. Resolve against the
/// installed library with [`skills_for_terms`].
pub fn project_terms(cwd: &str) -> Vec<String> {
    let mut out = Vec::new();
    if cwd.is_empty() {
        return out;
    }
    let mut dir = Some(Path::new(cwd));
    for _ in 0..PROJECT_WALK_LEVELS {
        let Some(d) = dir else { break };
        for (manifest, terms) in MANIFEST_TERMS {
            if d.join(manifest).exists() {
                for t in terms.iter() {
                    push_term(&mut out, t);
                }
            }
        }
        dir = d.parent();
    }
    out
}

/// Ecosystem terms implied by code files referenced in `text` (a prompt and/or
/// recent-window turns), via [`ext_terms`]. Order-preserving, de-duplicated.
pub fn code_terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for (_, ext) in file_tokens(text) {
        if let Some(terms) = ext_terms(&ext) {
            for t in terms.iter() {
                push_term(&mut out, t);
            }
        }
    }
    out
}

/// Resolve ecosystem `terms` against the installed library: every index entry
/// whose keywords (which include its name tokens) or description mention a term
/// maps to that term. Returns skill id → the matched term (for evidence display;
/// the first matching term in `terms` order wins, so callers should pass the
/// most specific term first). Matching is token-exact after [`norm_token`]
/// normalization — "uv" matches a `uv` keyword or "uv" in the description prose,
/// never a substring — and deliberately generous beyond that: this feeds the
/// *ambient* channel, which stays cosine-gated in [`crate::rank::rank_all_ctx`],
/// so a spurious term match costs nothing unless the skill was already
/// near-plausible for the prompt.
pub fn skills_for_terms(terms: &[String], idx: &Index) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    if terms.is_empty() {
        return out;
    }
    let terms: Vec<String> = terms.iter().map(|t| norm_token(t)).collect();
    for e in &idx.skills {
        let mut toks: BTreeSet<String> = e
            .keywords
            .iter()
            .flat_map(|k| tokenize(k))
            .map(|t| norm_token(&t))
            .collect();
        toks.extend(match_tokens(&e.description));
        if let Some(term) = terms.iter().find(|t| toks.contains(*t)) {
            out.insert(e.id.clone(), term.clone());
        }
    }
    out
}

/// The most-recent `depth` prompts of `recent` (which is oldest-first), or all of
/// them when fewer. Empty when the window is disabled (`context_depth == 0`) or
/// there is nothing to use. Gated on depth alone — the window is shared by the
/// dense-vector and file channels, so it must stay available even when the dense
/// blend (`context_weight`) is off.
fn window<'a>(recent: &'a [String], cfg: &Config) -> &'a [String] {
    if cfg.context_depth == 0 || recent.is_empty() {
        return &[];
    }
    let take = recent.len().min(cfg.context_depth);
    &recent[recent.len() - take..]
}

/// Build a single context vector from the recent-prompt window: a recency-weighted
/// mean of the per-prompt embeddings (geometric decay, most-recent weight 1.0).
/// `None` when the feature is off or the window is empty. Embeds the whole window
/// in one batch. The result need not be normalized — [`crate::rank::cosine`]
/// normalizes both operands.
pub fn vector(
    embedder: &dyn Embedder,
    recent: &[String],
    cfg: &Config,
) -> anyhow::Result<Option<Vec<f32>>> {
    if cfg.context_weight <= 0.0 {
        return Ok(None); // dense blend off (the window may still serve other channels)
    }
    let win = window(recent, cfg);
    if win.is_empty() {
        return Ok(None);
    }
    let embs = embedder.embed(win, EmbedKind::Query)?;
    let Some(dim) = embs.first().map(|e| e.len()) else {
        return Ok(None);
    };
    let n = embs.len();
    let mut acc = vec![0.0f32; dim];
    let mut wsum = 0.0f32;
    for (i, e) in embs.iter().enumerate() {
        // `recent`/`win` are oldest-first, so the last entry is the most recent and
        // earns weight 1.0; each older turn is halved.
        let w = 0.5f32.powi((n - 1 - i) as i32);
        wsum += w;
        for (a, x) in acc.iter_mut().zip(e) {
            *a += w * x;
        }
    }
    if wsum > 0.0 {
        for a in acc.iter_mut() {
            *a /= wsum;
        }
    }
    Ok(Some(acc))
}

/// The reranker query for a prompt whose best stage-1 self-cosine is `prompt_top`.
/// The recent-window text is prepended (so the cross-encoder reads the
/// conversation, including any named file) when context applies this turn — either
/// the prompt is vague enough that [`crate::rank::context_weight`] is positive, or
/// a file was referenced (`file_present`) and the file channel is on. Otherwise the
/// bare prompt is returned unchanged, so a confident, file-free prompt is never
/// muddied by stale context.
pub fn rerank_query(
    prompt: &str,
    prompt_top: f32,
    recent: &[String],
    file_present: bool,
    cfg: &Config,
) -> String {
    let win = window(recent, cfg);
    if win.is_empty() {
        return prompt.to_string();
    }
    let by_vagueness = crate::rank::context_weight(prompt_top, cfg) > 0.0;
    let by_file = file_present && cfg.file_boost > 0.0;
    if !(by_vagueness || by_file) {
        return prompt.to_string();
    }
    format!("{}\n{}", win.join("\n"), prompt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::bow::BowEmbedder;

    fn on() -> Config {
        Config {
            context_depth: 2,
            context_weight: 0.3,
            vague_lo: 0.55,
            vague_hi: 0.65,
            ..Default::default()
        }
    }

    #[test]
    fn vector_none_when_disabled_or_empty() {
        let e = BowEmbedder::new();
        // Feature off.
        let off = Config::default();
        assert!(vector(&e, &["a".into()], &off).unwrap().is_none());
        // On, but no prompts.
        assert!(vector(&e, &[], &on()).unwrap().is_none());
    }

    #[test]
    fn vector_built_when_enabled() {
        let e = BowEmbedder::new();
        let v = vector(
            &e,
            &["set up pytest".into(), "now the other one".into()],
            &on(),
        )
        .unwrap()
        .expect("a vector");
        assert!(!v.is_empty());
    }

    #[test]
    fn vector_respects_depth() {
        // Depth 1 must embed only the most recent prompt: the result equals the
        // single-prompt embedding of "b", not a mix with "a".
        let e = BowEmbedder::new();
        let cfg = Config {
            context_depth: 1,
            ..on()
        };
        let got = vector(&e, &["a a a".into(), "b b b".into()], &cfg)
            .unwrap()
            .unwrap();
        let want = e
            .embed(&["b b b".into()], EmbedKind::Query)
            .unwrap()
            .remove(0);
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < 1e-6);
        }
    }

    #[test]
    fn rerank_query_enriches_only_when_vague() {
        let recent = vec!["set up pytest".to_string()];
        let cfg = on();
        // Vague prompt (low self-cosine), no file -> prepend context.
        let vague = rerank_query("now the other one", 0.50, &recent, false, &cfg);
        assert_eq!(vague, "set up pytest\nnow the other one");
        // Confident prompt (high self-cosine), no file -> bare prompt.
        let confident = rerank_query("now the other one", 0.90, &recent, false, &cfg);
        assert_eq!(confident, "now the other one");
    }

    #[test]
    fn rerank_query_enriches_for_file_even_when_confident() {
        // A named file justifies enrichment regardless of prompt vagueness, as long
        // as the file channel is on and the window exists.
        let recent = vec!["attached sales.xlsx".to_string()];
        let cfg = Config {
            file_boost: 0.2,
            ..on()
        };
        let got = rerank_query("clean it up", 0.90, &recent, true, &cfg);
        assert_eq!(got, "attached sales.xlsx\nclean it up");
        // File channel off -> no enrichment from the file signal.
        let off_file = Config {
            file_boost: 0.0,
            ..on()
        };
        assert_eq!(
            rerank_query("clean it up", 0.90, &recent, true, &off_file),
            "clean it up"
        );
    }

    #[test]
    fn rerank_query_bare_when_window_off() {
        // context_depth 0 -> empty window -> always the bare prompt.
        let recent = vec!["set up pytest".to_string()];
        let off = Config {
            context_depth: 0,
            ..Config::default()
        };
        assert_eq!(rerank_query("x", 0.10, &recent, true, &off), "x");
    }

    #[test]
    fn file_ids_maps_known_extensions() {
        let got = file_ids("please clean up sales_q3.xlsx and merge report.pdf");
        assert!(got.contains("xlsx"));
        assert!(got.contains("pdf"));
        // Spreadsheet-family extensions all map to xlsx.
        assert!(file_ids("here is data.csv").contains("xlsx"));
        assert!(file_ids("the deck.pptx").contains("pptx"));
        assert!(file_ids("cover_letter.docx").contains("docx"));
        // OpenDocument / iWork / legacy office formats route by document kind.
        assert!(file_ids("budget.ods").contains("xlsx"));
        assert!(file_ids("notes.pages").contains("docx"));
        assert!(file_ids("keynote talk.key").contains("pptx"));
        assert!(file_ids("memo.rtf").contains("docx"));
    }

    #[test]
    fn file_ids_ignores_image_and_notebook_extensions() {
        // No 1:1 skill identity -> deliberately unmapped (see `ext_skill` docs).
        assert!(file_ids("see chart.png and demo.gif").is_empty());
        assert!(file_ids("open analysis.ipynb").is_empty());
    }

    #[test]
    fn project_terms_maps_manifest_in_cwd_and_ancestors() {
        // Hermetic temp tree: <root>/uv.lock and a nested cwd two levels down.
        let root = std::env::temp_dir().join(format!(
            "ski-proj-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let nested = root.join("src").join("inner");
        std::fs::create_dir_all(&nested).unwrap();
        std::fs::write(root.join("uv.lock"), b"version = 1\n").unwrap();

        // Manifest in cwd itself: uv.lock implies uv (most specific, listed
        // first) then python — order matters for evidence attribution.
        let terms = project_terms(root.to_str().unwrap());
        assert_eq!(terms, ["uv", "python"], "{terms:?}");
        // Manifest found by walking up from a nested cwd.
        assert!(project_terms(nested.to_str().unwrap())
            .iter()
            .any(|t| t == "uv"));

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn project_terms_empty_when_no_manifest_or_blank_cwd() {
        assert!(project_terms("").is_empty());
        // A nonexistent path stats cleanly to empty.
        assert!(project_terms("/no/such/ski/path/here").is_empty());
    }

    #[test]
    fn code_terms_maps_referenced_code_files() {
        let got = code_terms("please fix scripts/etl.py and look at handler.rs");
        assert!(
            got.iter().any(|t| t == "python") && got.iter().any(|t| t == "rust"),
            "{got:?}"
        );
        // Document formats belong to the file channel, not this one; prose with no
        // filenames maps nothing.
        assert!(code_terms("clean up report.xlsx").is_empty());
        assert!(code_terms("set up a project").is_empty());
    }

    #[test]
    fn skills_for_terms_matches_installed_library_dynamically() {
        let entry = |id: &str, description: &str, keywords: &[&str]| crate::index::Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: description.to_string(),
            path: String::new(),
            keywords: keywords.iter().map(|k| k.to_string()).collect(),
            trigger_phrases: Vec::new(),
            hash: String::new(),
            embedding: Vec::new(),
        };
        let idx = crate::index::Index {
            model: "test".into(),
            dim: 0,
            skills: vec![
                // The user's-own-library case: matched via its `uv` keyword.
                entry(
                    "uv-development",
                    "Bootstrap and manage projects.",
                    &["uv", "python"],
                ),
                // Matched via description prose only (no keywords).
                entry(
                    "rusty-style",
                    "Idiomatic Rust patterns and error handling.",
                    &[],
                ),
                // No ecosystem mention anywhere -> unmatched.
                entry("git-attribution", "Credit AI assistance in commits.", &[]),
            ],
        };
        let terms = vec!["uv".to_string(), "python".to_string(), "rust".to_string()];
        let got = skills_for_terms(&terms, &idx);
        // First matching term wins: uv-development matches both `uv` and `python`
        // keywords, and reports the more specific, earlier-listed `uv`.
        assert_eq!(got.get("uv-development").map(String::as_str), Some("uv"));
        assert_eq!(got.get("rusty-style").map(String::as_str), Some("rust"));
        assert!(!got.contains_key("git-attribution"));
        // Empty terms resolve to nothing.
        assert!(skills_for_terms(&[], &idx).is_empty());
    }

    #[test]
    fn file_ids_ignores_unmapped_and_bare_extensions() {
        // Code files map to no skill; a bare ".pdf" with no stem is not a reference.
        assert!(file_ids("edit main.rs and lib.py").is_empty());
        assert!(file_ids("the .pdf format is great").is_empty());
        // Trailing prose punctuation does not defeat the match.
        assert!(file_ids("look at budget.xlsx, then stop.").contains("xlsx"));
    }
}
