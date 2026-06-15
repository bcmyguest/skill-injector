//! Stage-2 cross-encoder reranking, gated on stage-1 ambiguity.
//!
//! The bi-encoder (stage 1, [`crate::rank`]) embeds query and skill description
//! independently; its cosine scores pile into a muddy ~0.60 band where genuine
//! matches and noise overlap, and it is confidently wrong on confusable pairs
//! (canvas-design vs algorithmic-art, docx vs pdf). A cross-encoder reads the
//! (prompt, skill) pair *jointly* and separates them: real matches score high,
//! noise crashes well negative.
//!
//! It is far costlier than the bi-encoder (a second ONNX model load + inference
//! on the hot path), so [`is_ambiguous`] gates it: a confident lone winner, or a
//! prompt with nothing relevant, skips stage 2 entirely and pays nothing. Only
//! the murky middle reaches the reranker.
//!
//! Feature-gated: without `fastembed`, [`rerank`] returns `None` and the caller
//! keeps the stage-1 result — identical behaviour to before this stage existed.

use crate::config::Config;
use crate::index::Index;
use crate::rank::Hit;

/// Whether stage-1 results warrant the cross-encoder. Skip (return `false`) when:
/// - nothing clears the recall floor (the prompt has no relevant skill), or
/// - the top match is a confident lone winner: high absolute score *and* a clear
///   gap to the runner-up.
///
/// Everything else — clustered peers, or a match stuck in the muddy band — is
/// ambiguous and reranked. The gate is deliberately conservative (errs toward
/// reranking) because the bi-encoder is confidently wrong on exactly the
/// clustered cases, so only an unmistakable single winner is allowed to skip.
pub fn is_ambiguous(hits: &[Hit], cfg: &Config) -> bool {
    let Some(top) = hits.first() else {
        return false;
    };
    if top.score < cfg.recall_floor {
        return false; // nothing relevant; stage-1 floor rejects it anyway.
    }
    // Confidence is measured on *cosine*, not the keyword-inflated `score`: a
    // keyword boost (e.g. "commit" matching pre-commit-setup) can fake a high
    // score and a clear gap, but that is precisely the noisy signal the
    // cross-encoder exists to arbitrate, so it must never grant a rerank skip.
    let mut cos: Vec<f32> = hits.iter().map(|h| h.cosine).collect();
    cos.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let c1 = cos[0];
    let c2 = cos.get(1).copied().unwrap_or(0.0);
    if c1 >= cfg.high_conf && (c1 - c2) >= cfg.clear_gap {
        return false; // lone, confident winner.
    }
    true
}

/// Rerank the top-`cfg.rerank_top_k` stage-1 candidates with the cross-encoder,
/// returning them rescored on the reranker's (logit) scale and sorted descending.
/// `Some` only with the `fastembed` feature and a usable model; `None` otherwise,
/// so the caller falls back to the stage-1 ordering.
///
/// `cosine`/`keyword` on each returned [`Hit`] are preserved for display; `score`
/// is replaced by the reranker logit. Callers must gate the result with the
/// reranker thresholds ([`Config::rerank_min`] / [`Config::rerank_margin`]), not
/// the bi-encoder ones — the scales differ.
pub fn rerank(hits: &[Hit], idx: &Index, prompt: &str, cfg: &Config) -> Option<Vec<Hit>> {
    #[cfg(feature = "fastembed")]
    {
        fast::rerank(hits, idx, prompt, cfg)
    }
    #[cfg(not(feature = "fastembed"))]
    {
        let _ = (hits, idx, prompt, cfg);
        None
    }
}

/// Apply the reranker-scale guardrails to a reranked candidate list: keep hits at
/// or above `rerank_min` and within `rerank_margin` of the best reranked score.
/// Returns hits sorted by descending reranked score (input order is preserved as
/// it already is). The caller still applies deny/session/cap.
pub fn passes(reranked: &[Hit], cfg: &Config) -> Vec<Hit> {
    let best = reranked
        .first()
        .map(|h| h.score)
        .unwrap_or(f32::NEG_INFINITY);
    reranked
        .iter()
        .filter(|h| h.score >= cfg.rerank_min && h.score >= best - cfg.rerank_margin)
        .cloned()
        .collect()
}

#[cfg(feature = "fastembed")]
mod fast {
    use super::*;
    use crate::skill;
    use fastembed::{RerankInitOptions, RerankerModel, TextRerank};
    use std::path::Path;
    use std::sync::OnceLock;

    /// The reranker is expensive to construct; build it once per process. The hook
    /// is a short-lived process (one prompt), so this is effectively per-prompt,
    /// but `why`/tests that rerank many prompts pay the load only once.
    fn model() -> Option<&'static TextRerank> {
        static MODEL: OnceLock<Option<TextRerank>> = OnceLock::new();
        MODEL
            .get_or_init(|| {
                // JINA turbo: on the anthropic/skills corpus it beat bge-reranker
                // -base on both top-1 accuracy and latency (266ms vs 2.8s load).
                TextRerank::try_new(
                    RerankInitOptions::new(RerankerModel::JINARerankerV1TurboEn)
                        .with_cache_dir(crate::paths::model_cache_dir())
                        .with_show_download_progress(false),
                )
                .ok()
            })
            .as_ref()
    }

    /// Document text for a candidate: its description plus the body head, read from
    /// the skill's `SKILL.md`. Falls back to the indexed description if the file is
    /// gone or unparseable.
    fn doc_text(entry: &crate::index::Entry) -> String {
        skill::parse_file(Path::new(&entry.path))
            .ok()
            .flatten()
            .map(|s| s.doc_text())
            .unwrap_or_else(|| entry.description.clone())
    }

    pub fn rerank(hits: &[Hit], idx: &Index, prompt: &str, cfg: &Config) -> Option<Vec<Hit>> {
        let reranker = model()?;
        let cands: Vec<&Hit> = hits.iter().take(cfg.rerank_top_k).collect();
        if cands.is_empty() {
            return None;
        }
        let docs: Vec<String> = cands
            .iter()
            .map(|h| idx.get(&h.id).map(doc_text).unwrap_or_default())
            .collect();
        let results = reranker
            .rerank(prompt.to_string(), docs, false, None)
            .ok()?;
        // results are sorted desc by score; map each back to its candidate Hit.
        let out = results
            .into_iter()
            .map(|r| {
                let src = cands[r.index];
                Hit {
                    score: r.score,
                    ..src.clone()
                }
            })
            .collect();
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        Config {
            recall_floor: 0.45,
            high_conf: 0.75,
            clear_gap: 0.12,
            rerank_min: -2.5,
            rerank_margin: 2.0,
            rerank_top_k: 12,
            ..Default::default()
        }
    }

    fn hit(id: &str, score: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: score,
            keyword: 0.0,
            score,
        }
    }

    #[test]
    fn nothing_relevant_is_not_ambiguous() {
        // Best below the recall floor -> skip the reranker.
        assert!(!is_ambiguous(&[hit("a", 0.40), hit("b", 0.38)], &cfg()));
    }

    #[test]
    fn confident_lone_winner_is_not_ambiguous() {
        // High top, clear gap -> skip.
        assert!(!is_ambiguous(&[hit("a", 0.82), hit("b", 0.60)], &cfg()));
    }

    #[test]
    fn clustered_peers_are_ambiguous() {
        // High but close together -> rerank (the confusable case).
        assert!(is_ambiguous(&[hit("a", 0.80), hit("b", 0.78)], &cfg()));
    }

    #[test]
    fn muddy_band_is_ambiguous() {
        // Above recall floor but below high-confidence -> rerank.
        assert!(is_ambiguous(&[hit("a", 0.62), hit("b", 0.55)], &cfg()));
    }

    #[test]
    fn empty_is_not_ambiguous() {
        assert!(!is_ambiguous(&[], &cfg()));
    }

    #[test]
    fn passes_keeps_top_and_rejects_negatives() {
        // Reranker scale: a strong match, a co-relevant peer, and noise.
        let reranked = vec![hit("a", 1.10), hit("b", -0.30), hit("c", -3.90)];
        let got: Vec<String> = passes(&reranked, &cfg())
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]); // c below rerank_min, and outside margin
    }

    #[test]
    fn passes_drops_all_when_best_is_noise() {
        let reranked = vec![hit("a", -2.83), hit("b", -3.94)];
        assert!(passes(&reranked, &cfg()).is_empty()); // negative prompt -> nothing
    }
}
