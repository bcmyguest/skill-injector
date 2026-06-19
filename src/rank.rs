//! Hybrid ranking: cosine(query, skill-description) + context blend + file boost
//! + keyword boost + phrase boost.

use crate::config::Config;
use crate::index::Index;
use crate::text::{content_tokens, tokenize};
use std::collections::{BTreeSet, HashSet};

#[derive(Clone, Debug)]
pub struct Hit {
    pub id: String,
    pub name: String,
    /// Cosine of the *current prompt* against the skill — kept pure (never folded
    /// with the context blend) so confidence/agreement gates can still read the
    /// prompt's own signal.
    pub cosine: f32,
    /// Boost from conversational context (see [`rank_all_ctx`]). Zero when the
    /// context feature is off, the prompt is confident, or the skill is no more
    /// context-relevant than average. Kept separate from `cosine` for attribution.
    pub context: f32,
    /// Boost from a referenced file of this skill's type (see
    /// [`crate::context::file_ids`]). Zero unless a matching file was named in the
    /// prompt or recent context. Separate for attribution — the highest-precision,
    /// directly-attributable context signal.
    pub file: f32,
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

/// Effective context-blend weight for a prompt whose best self-match cosine is
/// `prompt_top`. Scales from `cfg.context_weight` (a *fully vague* prompt,
/// `prompt_top <= vague_lo`) down to `0` (a *confident* prompt,
/// `prompt_top >= vague_hi`), linearly between. So a specific prompt ignores
/// context — avoiding the redundancy that regressed bi-encoder mean-centering
/// (see `crate::rerank` module docs) — while a vague follow-up leans on it.
/// Returns `0` whenever the feature is disabled (`context_weight <= 0` or
/// `context_depth == 0`).
pub fn context_weight(prompt_top: f32, cfg: &Config) -> f32 {
    if cfg.context_weight <= 0.0 || cfg.context_depth == 0 {
        return 0.0;
    }
    let (lo, hi) = (cfg.vague_lo, cfg.vague_hi);
    let vagueness = if hi <= lo {
        // Degenerate band: a hard step at `hi`.
        if prompt_top >= hi {
            0.0
        } else {
            1.0
        }
    } else {
        ((hi - prompt_top) / (hi - lo)).clamp(0.0, 1.0)
    };
    cfg.context_weight * vagueness
}

/// All skills, scored and sorted by descending hybrid score. No threshold — for
/// `ski why` and as input to [`select`]. No conversational context.
pub fn rank_all(query: &[f32], prompt: &str, index: &Index, cfg: &Config) -> Vec<Hit> {
    rank_all_ctx(query, None, &BTreeSet::new(), prompt, index, cfg)
}

/// Like [`rank_all`], but blends an optional conversational-context vector into
/// each skill's score. The blend is gated by how *vague* the current prompt is
/// ([`context_weight`]) and is a *relative* signal: a skill is boosted only in
/// proportion to how much more context-relevant it is than the average skill
/// (`cos(context, skill) - mean`), clamped at 0. That self-normalization is what
/// keeps the anisotropic bge floor (every skill cosines ~0.5 to anything) from
/// uniformly inflating scores and manufacturing false injects.
///
/// `file_ids` carries the file-type channel: any skill whose id is in the set
/// (a file of its type was named in the prompt/context — see
/// [`crate::context::file_ids`]) gets a flat `cfg.file_boost`, *not* gated on
/// vagueness, since a named file is unambiguous. With `context = None`, an empty
/// `file_ids`, and the features disabled, this is identical to [`rank_all`].
pub fn rank_all_ctx(
    query: &[f32],
    context: Option<&[f32]>,
    file_ids: &BTreeSet<String>,
    prompt: &str,
    index: &Index,
    cfg: &Config,
) -> Vec<Hit> {
    // The prompt's own cosines; their max gauges prompt specificity, which sets
    // how much (if any) context is allowed to contribute.
    let prompt_cos: Vec<f32> = index
        .skills
        .iter()
        .map(|e| cosine(query, &e.embedding))
        .collect();
    let prompt_top = prompt_cos.iter().copied().fold(0.0_f32, f32::max);
    let lambda = match context {
        Some(_) => context_weight(prompt_top, cfg),
        None => 0.0,
    };

    // Context cosines and their mean (the relative-boost baseline), computed once.
    let ctx_cos: Vec<f32> = match (lambda > 0.0, context) {
        (true, Some(c)) => index
            .skills
            .iter()
            .map(|e| cosine(c, &e.embedding))
            .collect(),
        _ => Vec::new(),
    };
    let ctx_mean = if ctx_cos.is_empty() {
        0.0
    } else {
        ctx_cos.iter().sum::<f32>() / ctx_cos.len() as f32
    };

    let mut hits: Vec<Hit> = index
        .skills
        .iter()
        .enumerate()
        .map(|(i, e)| {
            let cosine = prompt_cos[i];
            let context = ctx_cos
                .get(i)
                .map(|&c| lambda * (c - ctx_mean).max(0.0))
                .unwrap_or(0.0);
            let file = if cfg.file_boost > 0.0 && file_ids.contains(&e.id) {
                cfg.file_boost
            } else {
                0.0
            };
            let keyword = keyword_score(prompt, &e.keywords, cfg.keyword_boost);
            let phrase = phrase_score(prompt, &e.trigger_phrases, cfg.phrase_boost);
            Hit {
                id: e.id.clone(),
                name: e.name.clone(),
                cosine,
                context,
                file,
                keyword,
                phrase,
                score: cosine + context + file + keyword + phrase,
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
    use crate::index::{Entry, Index};

    fn no_files() -> BTreeSet<String> {
        BTreeSet::new()
    }

    /// Context enabled with a known vague band, everything else default.
    fn ctx_cfg() -> Config {
        Config {
            context_depth: 1,
            context_weight: 0.3,
            vague_lo: 0.55,
            vague_hi: 0.65,
            file_boost: 0.0, // context-only baseline; file tests opt the channel in
            ..Default::default()
        }
    }

    fn idx2() -> Index {
        let entry = |id: &str, emb: Vec<f32>| Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            path: String::new(),
            keywords: Vec::new(),
            trigger_phrases: Vec::new(),
            hash: String::new(),
            embedding: emb,
        };
        Index {
            model: "m".into(),
            dim: 2,
            skills: vec![entry("a", vec![1.0, 0.0]), entry("b", vec![0.0, 1.0])],
        }
    }

    #[test]
    fn context_weight_scales_with_vagueness() {
        let cfg = ctx_cfg(); // lo 0.55, hi 0.65, weight 0.3
        assert!((context_weight(0.50, &cfg) - 0.30).abs() < 1e-6); // <= lo: full
        assert_eq!(context_weight(0.65, &cfg), 0.0); // >= hi: none
        assert!((context_weight(0.60, &cfg) - 0.15).abs() < 1e-6); // midpoint: half
    }

    #[test]
    fn context_weight_zero_when_disabled() {
        let off_weight = Config {
            context_depth: 1,
            context_weight: 0.0,
            ..Default::default()
        };
        let off_depth = Config {
            context_depth: 0,
            context_weight: 0.3,
            ..Default::default()
        };
        assert_eq!(context_weight(0.10, &off_weight), 0.0);
        assert_eq!(context_weight(0.10, &off_depth), 0.0);
    }

    #[test]
    fn context_none_matches_plain_rank() {
        // With no context vector, scores are exactly cosine+keyword+phrase and the
        // context term is zero — identical to the pre-feature path.
        let q = [0.5, 0.5];
        let hits = rank_all_ctx(&q, None, &no_files(), "", &idx2(), &ctx_cfg());
        for h in &hits {
            assert_eq!(h.context, 0.0);
            assert!((h.score - h.cosine).abs() < 1e-6);
        }
    }

    #[test]
    fn vague_prompt_lets_context_break_a_tie() {
        // Prompt sits symmetrically between the two skills (cosine 0.707 to each),
        // and is vague (top 0.707 >= hi 0.65 -> NOT vague). Widen the band so it
        // counts as vague, then context pointing at `a` lifts `a` above `b`.
        let cfg = Config {
            vague_lo: 0.80,
            vague_hi: 0.90,
            ..ctx_cfg()
        };
        let q = [0.5, 0.5]; // equal cosine to a and b
        let ctx = [1.0, 0.0]; // points at a
        let hits = rank_all_ctx(&q, Some(&ctx), &no_files(), "", &idx2(), &cfg);
        assert_eq!(hits[0].id, "a"); // context broke the tie
        assert!(hits[0].context > 0.0);
        // `b` is no more context-relevant than average, so it gets no boost.
        let b = hits.iter().find(|h| h.id == "b").unwrap();
        assert_eq!(b.context, 0.0);
    }

    #[test]
    fn confident_prompt_suppresses_context() {
        // Prompt is exactly `a` (cosine 1.0 >= hi): context (pointing at `b`) must
        // not contribute, so no skill carries a context boost.
        let q = [1.0, 0.0];
        let ctx = [0.0, 1.0];
        let hits = rank_all_ctx(&q, Some(&ctx), &no_files(), "", &idx2(), &ctx_cfg());
        assert!(hits.iter().all(|h| h.context == 0.0));
        assert_eq!(hits[0].id, "a");
    }

    #[test]
    fn file_boost_lifts_named_skill_ungated() {
        // A referenced file boosts its skill even when the prompt is *confident*
        // about a different skill (file channel is not vagueness-gated). Prompt is
        // exactly `a`; a file of `b`'s type is named.
        let cfg = Config {
            file_boost: 0.2,
            ..ctx_cfg()
        };
        let q = [1.0, 0.0]; // confident about `a`
        let files: BTreeSet<String> = ["b".to_string()].into_iter().collect();
        let hits = rank_all_ctx(&q, None, &files, "", &idx2(), &cfg);
        let b = hits.iter().find(|h| h.id == "b").unwrap();
        assert!((b.file - 0.2).abs() < 1e-6); // b carries the file boost
        let a = hits.iter().find(|h| h.id == "a").unwrap();
        assert_eq!(a.file, 0.0); // a does not (no file of its type)
    }

    #[test]
    fn file_boost_off_when_zero() {
        let q = [1.0, 0.0];
        let files: BTreeSet<String> = ["b".to_string()].into_iter().collect();
        // file_boost defaults to 0.0 in ctx_cfg -> no file term anywhere.
        let hits = rank_all_ctx(&q, None, &files, "", &idx2(), &ctx_cfg());
        assert!(hits.iter().all(|h| h.file == 0.0));
    }

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
