//! Hybrid ranking: cosine(query, skill-description) + keyword boost.

use crate::config::Config;
use crate::index::Index;
use crate::text::tokenize;
use std::collections::HashSet;

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: String,
    pub name: String,
    pub cosine: f32,
    pub keyword: f32,
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

/// All skills, scored and sorted by descending hybrid score. No threshold — for
/// `ski why` and as input to [`select`].
pub fn rank_all(query: &[f32], prompt: &str, index: &Index, cfg: &Config) -> Vec<Hit> {
    let mut hits: Vec<Hit> = index
        .skills
        .iter()
        .map(|e| {
            let cosine = cosine(query, &e.embedding);
            let keyword = keyword_score(prompt, &e.keywords, cfg.keyword_boost);
            Hit {
                id: e.id.clone(),
                name: e.name.clone(),
                cosine,
                keyword,
                score: cosine + keyword,
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
}
