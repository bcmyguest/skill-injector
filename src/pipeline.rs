//! The shared decision pipeline: stage selection + gating, single-sourced so the
//! hook (hot path), `ski why` (tuning aid), and `examples/eval` run identical math.
//!
//! Previously each of the three re-implemented the stage cascade. `ski why` had
//! drifted the furthest — it ranked with [`crate::rank::rank_all`] (no
//! context/file/project channels) and reranked the *bare* prompt, so it could star
//! a different skill than the hook would inject on the same prompt. The tool meant
//! to explain the ranker didn't reproduce it.
//!
//! The caller owns the inputs that depend on conversation state (the query vector,
//! the context blend, file/project channels, the rerank query); this module owns
//! the cascade over the resulting `hits`:
//! 1. a dominant lexical (BM25) winner injects directly, unless stage-1 already has
//!    a confident lone dense winner;
//! 2. otherwise the cross-encoder arbitrates the ambiguous middle;
//! 3. otherwise the cheap stage-1 cosine result stands.
//!
//! It returns the winning stage, its rows (for display), and the hits that clear
//! that stage's gate — *before* deny / session-dedup / slash-removal / `max_skills`,
//! which stay with the caller (only the hook has a session).

use crate::confidence::Stage;
use crate::config::Config;
use crate::index::Index;
use crate::lexical::{self, Lex};
use crate::rank::Hit;
use crate::rerank;

/// The outcome of the decision cascade for one prompt.
#[derive(Debug)]
pub struct Plan {
    /// Which stage produced the decision.
    pub stage: Stage,
    /// The winning stage's ranking, for display: the reranked list when the
    /// cross-encoder fired, otherwise the stage-1 hits. (For the lexical fast-path
    /// these are still the stage-1 hits; the winner is in [`Plan::lexical`].)
    pub rows: Vec<Hit>,
    /// The lexical fast-path winner, if one fired.
    pub lexical: Option<Lex>,
    /// Hits that clear the winning stage's gate, in rank order, *before*
    /// deny / dedup / slash-removal / cap. For the lexical stage this is the single
    /// dominant winner (its stage-1 [`Hit`], pulled from `rows`).
    pub passed: Vec<Hit>,
    /// Display threshold for the winning stage (`min_similarity` / `rerank_min` /
    /// `lexical_min`).
    pub threshold: f32,
}

/// Stage-1 cosine gate: the hits clearing the absolute floor (`min_similarity`)
/// and within the relative margin (`score_margin`) of the leader, plus any `force`d
/// skill on a keyword hit. Pre deny/dedup/cap; pure (no IO), so it is the unit-test
/// seam for the gate the hook used to inline in `select`.
pub fn cosine_passed(hits: &[Hit], cfg: &Config) -> Vec<Hit> {
    let top = hits.first().map(|h| h.score).unwrap_or(0.0);
    hits.iter()
        .filter(|h| {
            let forced = cfg.force.contains(&h.id) && h.keyword > 0.0;
            forced || (h.score >= cfg.min_similarity && h.score >= top - cfg.score_margin)
        })
        .cloned()
        .collect()
}

/// Run the stage cascade over already-ranked `hits` (from
/// [`crate::rank::rank_all_ctx`]). `prompt` is the bare user prompt (for the lexical
/// channel); `rerank_query` is the context-enriched query the cross-encoder reads.
pub fn decide(hits: &[Hit], idx: &Index, prompt: &str, rerank_query: &str, cfg: &Config) -> Plan {
    // Stage 1.5: a dominant lexical (BM25-over-description) winner injects directly
    // — high precision exactly where the bi-encoder cosine is muddy — unless stage-1
    // already has a confident lone dense winner, which is trusted outright.
    if !rerank::confident_winner(hits, cfg) {
        if let Some(win) = lexical::dominant(prompt, idx, cfg) {
            let passed = hits.iter().filter(|h| h.id == win.id).cloned().collect();
            return Plan {
                stage: Stage::Lexical,
                rows: hits.to_vec(),
                lexical: Some(win),
                passed,
                threshold: cfg.lexical_min,
            };
        }
    }
    // Stage 2: the cross-encoder arbitrates the ambiguous middle; a confident winner
    // / nothing-relevant keeps the cheap stage-1 result.
    match rerank::is_ambiguous(hits, cfg)
        .then(|| rerank::rerank(hits, idx, rerank_query, cfg))
        .flatten()
    {
        Some(reranked) => {
            let passed = rerank::passes(&reranked, cfg);
            Plan {
                stage: Stage::Rerank,
                rows: reranked,
                lexical: None,
                passed,
                threshold: cfg.rerank_min,
            }
        }
        None => Plan {
            stage: Stage::Cosine,
            passed: cosine_passed(hits, cfg),
            rows: hits.to_vec(),
            lexical: None,
            threshold: cfg.min_similarity,
        },
    }
}

/// Human-readable stage label for `ski why` / `examples/eval` display.
/// `model` names the stage-1 embedder (only shown for the cosine stage).
pub fn stage_label(stage: Stage, model: &str) -> String {
    match stage {
        Stage::Cosine => format!("stage1:{model}"),
        Stage::Rerank => "rerank:turbo".to_string(),
        Stage::Lexical => "lexical(BM25)".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(id: &str, score: f32, keyword: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: score - keyword,
            context: 0.0,
            file: 0.0,
            project: 0.0,
            keyword,
            phrase: 0.0,
            score,
        }
    }

    #[test]
    fn cosine_passed_applies_floor_and_margin() {
        let cfg = Config::default(); // min 0.30, margin 0.15
        let hits = vec![
            hit("a", 0.90, 0.0),
            hit("b", 0.80, 0.0), // within margin of 0.90
            hit("c", 0.50, 0.0), // below 0.90 - 0.15 margin
            hit("d", 0.10, 0.0), // below the floor
        ];
        let got: Vec<String> = cosine_passed(&hits, &cfg)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]);
    }

    #[test]
    fn cosine_passed_force_bypasses_floor_on_keyword() {
        let cfg = Config {
            force: vec!["x".to_string()],
            ..Default::default()
        };
        // x is sub-floor but forced with a keyword hit; y is sub-floor, not forced.
        let hits = vec![hit("x", 0.10, 0.15), hit("y", 0.20, 0.0)];
        let got: Vec<String> = cosine_passed(&hits, &cfg)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["x"]);
    }

    #[test]
    fn stage_label_renders_each_stage() {
        assert_eq!(stage_label(Stage::Cosine, "bge"), "stage1:bge");
        assert_eq!(stage_label(Stage::Rerank, "bge"), "rerank:turbo");
        assert_eq!(stage_label(Stage::Lexical, "bge"), "lexical(BM25)");
    }
}
