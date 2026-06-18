//! Hybrid ranking: cosine(query, skill-description) + keyword boost + phrase boost.

use crate::config::Config;
use crate::index::Index;
use crate::text::{content_tokens, tokenize};
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: String,
    pub name: String,
    pub cosine: f32,
    pub keyword: f32,
    /// Boost from matched trigger phrases (see [`phrase_score`]).
    pub phrase: f32,
    pub score: f32,
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0f32, 0f32, 0f32);
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

pub fn keyword_score(prompt: &str, keywords: &[String], boost: f32) -> f32 {
    let toks: HashSet<String> = tokenize(prompt).into_iter().collect();
    let hits = keywords
        .iter()
        .filter(|k| toks.contains(k.as_str()))
        .count();
    hits as f32 * boost
}

/// Phrase channel: `boost` per trigger phrase whose every content token appears in
/// the prompt. A phrase is the normalized (content-token) form produced by
/// [`crate::skill::extract_phrases`]; requiring *all* its tokens (>=2 by
/// construction) keeps the signal high-precision, so it lifts a skill on the
/// exact wording the bi-encoder dilutes without firing on incidental overlap.
pub fn phrase_score(prompt: &str, phrases: &[String], boost: f32) -> f32 {
    if phrases.is_empty() {
        return 0.0;
    }
    let toks: HashSet<String> = content_tokens(prompt).into_iter().collect();
    let hits = phrases
        .iter()
        .filter(|p| {
            let mut pt = p.split_whitespace().peekable();
            pt.peek().is_some() && pt.all(|t| toks.contains(t))
        })
        .count();
    hits as f32 * boost
}

/// All skills, scored and sorted by descending hybrid score. No threshold — for
/// `ski why` and as input to [`select`].
pub fn rank_all(query: &[f32], prompt: &str, index: &Index, cfg: &Config) -> Vec<Hit> {
    let mut hits: Vec<Hit> = index
        .skills
        .iter()
        .map(|e| {
            let cosine = cosine(query, &e.embedding);
            let keyword = keyword_score(prompt, &e.keywords, cfg.keyword_boost);
            let phrase = phrase_score(prompt, &e.trigger_phrases, cfg.phrase_boost);
            Hit {
                id: e.id.clone(),
                name: e.name.clone(),
                cosine,
                keyword,
                phrase,
                score: cosine + keyword + phrase,
            }
        })
        .collect();
    hits.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    hits
}

/// Apply the injection guardrails: drop below `min_similarity`, cap at `max_skills`.
pub fn select(hits: Vec<Hit>, cfg: &Config) -> Vec<Hit> {
    hits.into_iter()
        .filter(|h| h.score >= cfg.min_similarity)
        .take(cfg.max_skills)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_bounds() {
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0];
        let c = [0.0, 1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn keyword_boost_counts_matches() {
        let kw = vec!["uv".to_string(), "setup".to_string()];
        assert!((keyword_score("set up with uv", &kw, 0.1) - 0.1).abs() < 1e-6); // only "uv"
        assert!((keyword_score("uv setup now", &kw, 0.1) - 0.2).abs() < 1e-6); // both
    }

    #[test]
    fn phrase_fires_only_when_all_tokens_present() {
        let ph = vec!["screen reader support".to_string()];
        // Full phrase present (any order, extra words around) -> boost.
        assert!(
            (phrase_score("does my form have screen reader support today", &ph, 0.2) - 0.2).abs()
                < 1e-6
        );
        // Reordered, still all tokens present -> boost.
        assert!((phrase_score("support for a screen reader", &ph, 0.2) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn phrase_does_not_fire_on_partial_overlap() {
        // Precision guard: a partial token overlap must NOT boost, or the phrase
        // channel would manufacture false positives on unrelated prompts.
        let ph = vec!["screen reader support".to_string()];
        assert_eq!(
            phrase_score("split this screen into two panes", &ph, 0.2),
            0.0
        );
        assert_eq!(
            phrase_score(
                "implement a debounce function in vanilla javascript",
                &ph,
                0.2
            ),
            0.0
        );
    }

    #[test]
    fn phrase_score_sums_distinct_phrases() {
        // Phrases are stored already normalized to content tokens (no stopwords),
        // the form `extract_phrases` produces.
        let ph = vec![
            "convert markdown pdf".to_string(),
            "merge two pdf files".to_string(),
        ];
        assert!(
            (phrase_score(
                "convert this markdown to pdf and merge two pdf files",
                &ph,
                0.2
            ) - 0.4)
                .abs()
                < 1e-6
        );
    }
}
