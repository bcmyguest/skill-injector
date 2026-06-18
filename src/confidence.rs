//! Map a stage score onto one `[0,1]` confidence axis plus a coarse band, shared
//! by the injection phrasing (how forcefully to recommend) and session dedup
//! (whether a *re*-recommendation clears the HIGH bar).
//!
//! Two scales reach us: stage-1 cosine (`~0.3–0.9`, anisotropic — unrelated
//! prompts still sit ~0.5) and stage-2 reranker logits (`~-10..+10`). They are
//! not comparable, so each gets its own mapping. The reranker mapping is
//! principled (a sigmoid, matching the cross-encoder's training objective); the
//! cosine mapping is an explicit heuristic — it exists so phrasing/dedup have a
//! single dial, not to claim cosine is a probability.

use crate::config::Config;

/// Which ranking stage produced a score, selecting its confidence mapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Stage-1 bi-encoder cosine (+ keyword boost).
    Cosine,
    /// Stage-2 cross-encoder reranker logit.
    Rerank,
}

/// Coarse confidence band, driving phrasing forcefulness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Band {
    High,
    Medium,
    Low,
}

/// `>=` this is the High band, and the bar a *repeat* recommendation must clear
/// (see [`crate::session::Session::should_recommend`]).
pub const HIGH: f32 = 0.80;
/// `<` this is the Low (tentative) band.
pub const LOW: f32 = 0.55;
/// Cosine span above `min_similarity` over which confidence climbs floor->ceiling.
/// Heuristic: bge's genuinely-strong matches sit roughly this far above the
/// eligibility floor.
const COSINE_SPAN: f32 = 0.45;

/// Confidence in `[0,1]` for a hit's `score`, given the stage that produced it.
pub fn of(score: f32, stage: Stage, cfg: &Config) -> f32 {
    match stage {
        // JINA-turbo logits are ~calibrated; sigmoid -> probability.
        Stage::Rerank => sigmoid(score),
        // Cosine has no probabilistic meaning; map [floor, floor+span] -> [.5,.97].
        Stage::Cosine => {
            let t = ((score - cfg.min_similarity) / COSINE_SPAN).clamp(0.0, 1.0);
            (0.5 + 0.47 * t).clamp(0.0, 0.99)
        }
    }
}

/// Band for a confidence value.
pub fn band(conf: f32) -> Band {
    if conf >= HIGH {
        Band::High
    } else if conf >= LOW {
        Band::Medium
    } else {
        Band::Low
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn rerank_sigmoid_anchors() {
        let cfg = Config::default();
        assert!((of(0.0, Stage::Rerank, &cfg) - 0.5).abs() < 1e-3);
        assert!(of(3.0, Stage::Rerank, &cfg) > 0.9); // strong match
        assert!(of(-2.5, Stage::Rerank, &cfg) < 0.1); // at the rerank floor
    }

    #[test]
    fn cosine_climbs_from_floor() {
        let cfg = Config::default(); // min_similarity 0.30
        let at_floor = of(0.30, Stage::Cosine, &cfg);
        let strong = of(0.80, Stage::Cosine, &cfg);
        assert!((at_floor - 0.5).abs() < 1e-3);
        assert!(strong > HIGH);
        assert!(strong <= 0.99);
    }

    #[test]
    fn cosine_clamps_below_floor() {
        let cfg = Config::default();
        // A forced sub-floor keyword hit must not produce a negative confidence.
        assert!(of(0.0, Stage::Cosine, &cfg) >= 0.0);
    }

    #[test]
    fn bands_partition_the_axis() {
        assert_eq!(band(0.95), Band::High);
        assert_eq!(band(HIGH), Band::High);
        assert_eq!(band(0.70), Band::Medium);
        assert_eq!(band(LOW), Band::Medium);
        assert_eq!(band(0.40), Band::Low);
    }
}
