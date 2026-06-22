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
//!
//! **Rejected experiment — mean-centering the bi-encoder space.** The classic
//! anisotropy fix (subtract the corpus-mean embedding from the query and every
//! skill vector before cosine, then renormalize) was implemented and measured
//! against `examples/eval` across all three fixtures. It *did* sharpen stage 1 —
//! stage-1 top-1 rose (e.g. 75% -> 84% on the anthropic set) and recall@`rerank_top_k`
//! went 98% -> 100% (it recovered the one true retrieval miss) — but the final,
//! post-rerank recall *regressed* ~3 points (93/106 -> 91/106) at equal false-inject,
//! across a min_similarity sweep. The reason is the finding `examples/eval`'s
//! recall@k instrumentation made explicit: retrieval is not the bottleneck (gold is
//! almost always already in the top-k), so a sharper bi-encoder is largely redundant
//! with this reranker, while the shifted cosine distribution disrupts the gate it
//! feeds. Not worth the added complexity, the new persisted `mean`, and the forced
//! reindex. Revisit only if the reranker is removed or the live distribution proves
//! materially different from the eval corpus.

use crate::config::Config;
use crate::index::Index;
use crate::rank::Hit;

/// How far below stage-1's solo-injection floor (`min_similarity`) a reranked
/// candidate may sit and still inject — the cosine "credit" the cross-encoder's
/// confirmation is worth. Tuned on the realistic corpus *and* a live 56-skill
/// library: a borderline real match the bi-encoder ranks at ~0.63 ("clean up this
/// messy CSV" -> xlsx, cosine 0.634) injects, while the false-inject skills cluster
/// lower (~0.57-0.59) and stay out. Sweep: at this slack recall holds 95% / false
/// injects 2%; a smaller slack (floor 0.64) drops a positive, a larger one (0.58)
/// readmits an FP. See `examples/eval`.
const AGREEMENT_SLACK: f32 = 0.03;

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
    // A confident file-channel leader is trusted without the cross-encoder. A named
    // file (`.xlsx`/`.pdf`/`.docx`/...) is a high-precision, directly-attributable
    // signal — unlike the keyword boost this gate deliberately excludes below — so
    // when it lifts its skill to a clear stage-1 lead, the reranker (which reads only
    // text) must not veto it on a vague follow-up like "pull the text out of it",
    // where the prompt itself carries almost no signal. Gated on a clear score gap so
    // a file merely mentioned in passing, with no dominant match, still reranks.
    if top.file > 0.0 {
        let s2 = hits.get(1).map(|h| h.score).unwrap_or(0.0);
        if top.score - s2 >= cfg.clear_gap {
            return false; // confident file-channel winner.
        }
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
///
/// **Stage-1 agreement.** Before the reranker thresholds, a candidate must have a
/// bi-encoder score (the preserved stage-1 `cosine + context + file + keyword +
/// phrase`; [`rerank`] only overwrites `score` with the logit) within [`AGREEMENT_SLACK`] of
/// stage-1's own injection floor (`min_similarity`). The phrase term is included on
/// purpose: a confident multi-token trigger match is exactly the "stage-1 judged
/// relevant" signal this gate looks for, so it may carry an otherwise sub-floor
/// cosine through. The context term rides along for the same reason — a vague prompt
/// the conversation made relevant is a stage-1 "relevant" signal too. The
/// cross-encoder's job is to reorder and confirm the *retrieved* relevant set, not
/// to resurrect a skill stage-1 judged irrelevant. Without this gate a prompt with no real match — "implement the
/// builder pattern in Java", "RSA key generation from scratch" — lets the reranker
/// pull a sub-floor skill to the top and inject noise; the logits there interleave
/// with genuine weak matches (so no `rerank_min` value separates them), but their
/// stage-1 scores sit lower (~0.57-0.59 vs ~0.63 for borderline real matches).
pub fn passes(reranked: &[Hit], cfg: &Config) -> Vec<Hit> {
    let floor = cfg.min_similarity - AGREEMENT_SLACK;
    // Keep only candidates stage-1 also rated relevant; the best *eligible* logit
    // then anchors the relative margin (a sub-floor leader can't drag peers in).
    let eligible: Vec<&Hit> = reranked
        .iter()
        .filter(|h| h.cosine + h.context + h.file + h.keyword + h.phrase >= floor)
        .collect();
    let best = eligible
        .first()
        .map(|h| h.score)
        .unwrap_or(f32::NEG_INFINITY);
    eligible
        .into_iter()
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
                // JINA turbo: on a realistic ~48-skill index it ties the 7x-larger
                // bge-reranker-base and jina-v2-base on top-1 accuracy and false-
                // injection rate, at a fraction of the load/latency cost. The gate
                // (`rerank_min`), not reranker size, is what controls noise here.
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

    /// For `is_ambiguous` tests, which read `cosine`: model a hit whose cosine is
    /// its score (no keyword boost).
    fn hit(id: &str, score: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: score,
            context: 0.0,
            file: 0.0,
            project: 0.0,
            keyword: 0.0,
            phrase: 0.0,
            score,
        }
    }

    /// For `passes` tests, which gate on the reranker *logit* (`score`) while the
    /// new stage-1-agreement filter reads `cosine`: keep them independent.
    fn rhit(id: &str, logit: f32, cosine: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine,
            context: 0.0,
            file: 0.0,
            project: 0.0,
            keyword: 0.0,
            phrase: 0.0,
            score: logit,
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

    /// A confident file-channel leader skips the reranker even though its raw cosine
    /// sits in the muddy band: the file boost is high-precision, so a clear lead is
    /// trusted. Mirrors the doc follow-ups ("pull the text out of it" -> pdf) the
    /// reranker was wrongly vetoing.
    #[test]
    fn confident_file_leader_is_not_ambiguous() {
        let lead = Hit {
            file: 0.30,
            score: 0.866, // cosine 0.566 + file 0.30
            ..hit("pdf", 0.566)
        };
        let runner = hit("pptx", 0.55);
        assert!(!is_ambiguous(&[lead, runner], &cfg()));
    }

    /// A file mentioned in passing with no dominant match (small score gap) still
    /// reranks — the boost alone must not wave a contested field through.
    #[test]
    fn weak_file_leader_still_reranks() {
        let lead = Hit {
            file: 0.30,
            score: 0.62, // cosine 0.32 + file 0.30
            ..hit("pdf", 0.32)
        };
        let runner = Hit {
            score: 0.58,
            ..hit("docx", 0.58)
        };
        assert!(is_ambiguous(&[lead, runner], &cfg())); // gap 0.04 < clear_gap 0.12
    }

    #[test]
    fn passes_keeps_top_and_rejects_negatives() {
        // Reranker scale: a strong match, a co-relevant peer, and noise. All three
        // cleared stage-1 (cosine above the 0.30 default floor), so only the logit
        // gates apply.
        let reranked = vec![
            rhit("a", 1.10, 0.80),
            rhit("b", -0.30, 0.70),
            rhit("c", -3.90, 0.65),
        ];
        let got: Vec<String> = passes(&reranked, &cfg())
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]); // c below rerank_min, and outside margin
    }

    #[test]
    fn passes_drops_all_when_best_is_noise() {
        let reranked = vec![rhit("a", -2.83, 0.70), rhit("b", -3.94, 0.66)];
        assert!(passes(&reranked, &cfg()).is_empty()); // negative prompt -> nothing
    }

    #[test]
    fn passes_rejects_subfloor_stage1_resurrection() {
        // The reranker pulled a skill to the top (high logit) whose stage-1 score
        // (cosine 0.20) sits well below the agreement floor (min_similarity 0.30
        // minus the slack): it must not be injected, even though its logit clears
        // `rerank_min`. This is the over-injection the builder-pattern / RSA
        // negatives produced.
        let reranked = vec![rhit("ghost", 1.50, 0.20), rhit("real", 0.40, 0.72)];
        let got: Vec<String> = passes(&reranked, &cfg())
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["real"]); // ghost dropped on stage-1 disagreement
    }

    #[test]
    fn passes_subfloor_leader_does_not_drag_in_peers() {
        // A sub-floor leader is dropped, and the relative margin is then anchored on
        // the best *eligible* skill — a trailing real skill outside the leader's
        // margin is still judged on its own.
        let cfg = cfg(); // rerank_margin 2.0
        let reranked = vec![rhit("ghost", 2.00, 0.20), rhit("real", -0.40, 0.72)];
        let got: Vec<String> = passes(&reranked, &cfg).into_iter().map(|h| h.id).collect();
        assert_eq!(got, ["real"]); // kept: -0.40 >= rerank_min, anchors its own margin
    }
}
