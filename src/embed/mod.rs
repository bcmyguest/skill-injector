//! Embedding backends behind a single trait.
//!
//! - [`bow::BowEmbedder`] — deterministic hashed bag-of-words. No deps, no
//!   network, no model. Always available; used for tests and offline fallback.
//! - `fast::FastEmbedder` — real bge-small / MiniLM via fastembed (ONNX). Behind
//!   the `fastembed` cargo feature.

pub mod bow;
#[cfg(feature = "fastembed")]
pub mod fast;

/// Whether a text is a search query or an indexed document. bge models are
/// asymmetric (query gets an instruction prefix); symmetric models ignore this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedKind {
    Query,
    Document,
}

pub trait Embedder {
    /// Stable id used as the index's `model` tag (changing it forces reindex).
    fn id(&self) -> String;
    fn embed(&self, texts: &[String], kind: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>>;

    /// Score floor under which the top match is treated as "nothing relevant"
    /// and the hook injects nothing. Embedder-specific because cosine
    /// distributions differ sharply: the hashed bag-of-words space is sparse, so
    /// unrelated text scores near 0 and a low floor works; bge is anisotropic, so
    /// even unrelated text cosines ~0.5 and the floor must sit well above that.
    /// The default is calibrated for bag-of-words; embedders override it.
    fn min_similarity(&self) -> f32 {
        0.30
    }

    /// Max score gap below the single best match for a co-relevant peer to still
    /// be injected. Tighter spaces (bge) need a smaller margin. Default is for
    /// bag-of-words.
    fn score_margin(&self) -> f32 {
        0.15
    }
}

/// Pick a backend for `model`. With the `fastembed` feature and a recognized
/// model id, returns the real embedder; otherwise the offline bag-of-words one.
pub fn build(model: &str) -> anyhow::Result<Box<dyn Embedder>> {
    #[cfg(feature = "fastembed")]
    {
        if let Some(e) = fast::FastEmbedder::try_for(model)? {
            return Ok(Box::new(e));
        }
    }
    let _ = model;
    Ok(Box::new(bow::BowEmbedder::new()))
}
