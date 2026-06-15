//! Deterministic hashed bag-of-words embedder.
//!
//! Not semantic (no synonyms) — surface-token overlap only. It exists so the
//! whole pipeline builds, runs, and tests with zero network/model, and as a
//! fallback on machines without the fastembed feature. The hybrid keyword boost
//! in the ranker compensates for its lack of semantics on exact terms.

use crate::embed::{EmbedKind, Embedder};
use crate::text::{fnv1a_32, tokenize};

pub struct BowEmbedder {
    dim: usize,
}

impl BowEmbedder {
    pub fn new() -> Self {
        Self { dim: 256 }
    }

    fn one(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0f32; self.dim];
        for tok in tokenize(text) {
            let idx = (fnv1a_32(&tok) as usize) % self.dim;
            v[idx] += 1.0;
        }
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

impl Default for BowEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for BowEmbedder {
    fn id(&self) -> String {
        format!("bow-{}-v1", self.dim)
    }

    fn embed(&self, texts: &[String], _kind: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.one(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_normalized() {
        let e = BowEmbedder::new();
        let a = &e
            .embed(&["commit this diff".into()], EmbedKind::Query)
            .unwrap()[0];
        let b = &e
            .embed(&["commit this diff".into()], EmbedKind::Document)
            .unwrap()[0];
        assert_eq!(a, b);
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn overlap_scores_higher_than_disjoint() {
        let e = BowEmbedder::new();
        let q = &e
            .embed(&["python project setup".into()], EmbedKind::Query)
            .unwrap()[0];
        let near = &e
            .embed(&["set up a python project".into()], EmbedKind::Document)
            .unwrap()[0];
        let far = &e
            .embed(&["lemonade server gpu".into()], EmbedKind::Document)
            .unwrap()[0];
        let cos = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        assert!(cos(q, near) > cos(q, far));
    }
}
