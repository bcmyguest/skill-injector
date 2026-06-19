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
use std::collections::BTreeSet;

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

/// Skill ids implied by file references in `text` (a prompt and/or recent-window
/// turns): scans whitespace-separated tokens for a trailing `.<ext>` and maps each
/// known extension through [`ext_skill`]. This is the *directly attributable*
/// context signal — a file attached or named **now** is unambiguous in a way a
/// vague prompt is not, so unlike the dense context vector it is not gated on
/// prompt vagueness. De-duplicated.
pub fn file_ids(text: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for tok in
        text.split(|c: char| c.is_whitespace() || matches!(c, '"' | '\'' | '(' | ')' | '`' | ','))
    {
        // Strip trailing punctuation that commonly hugs a filename in prose.
        let tok = tok.trim_end_matches(['.', ':', ';', '!', '?']);
        if let Some((stem, ext)) = tok.rsplit_once('.') {
            if stem.is_empty() {
                continue; // a bare ".pdf" with no name is not a real reference.
            }
            let ext = ext.to_ascii_lowercase();
            if let Some(id) = ext_skill(&ext) {
                out.insert(id.to_string());
            }
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
    fn file_ids_ignores_unmapped_and_bare_extensions() {
        // Code files map to no skill; a bare ".pdf" with no stem is not a reference.
        assert!(file_ids("edit main.rs and lib.py").is_empty());
        assert!(file_ids("the .pdf format is great").is_empty());
        // Trailing prose punctuation does not defeat the match.
        assert!(file_ids("look at budget.xlsx, then stop.").contains("xlsx"));
    }
}
