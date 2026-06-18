//! Real embeddings via fastembed (ONNX). Compiled with the `fastembed` feature,
//! which is on by default; build `--no-default-features` to drop it for the offline
//! bag-of-words lane. Default model: bge-small-en-v1.5; lite alt: all-MiniLM-L6-v2
//! (quantized).

use crate::embed::{EmbedKind, Embedder};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const BGE_QUERY_PREFIX: &str = "Represent this sentence for searching relevant passages: ";

pub struct FastEmbedder {
    model: TextEmbedding,
    tag: String,
    bge: bool,
}

impl FastEmbedder {
    /// `Some` if `model` is a recognized fastembed model id, else `None` so the
    /// caller can fall back to the bag-of-words embedder.
    pub fn try_for(model: &str) -> anyhow::Result<Option<Self>> {
        let (em, bge) = match model {
            "bge-small-en-v1.5" => (EmbeddingModel::BGESmallENV15, true),
            "bge-base-en-v1.5" => (EmbeddingModel::BGEBaseENV15, true),
            "all-MiniLM-L6-v2-q" => (EmbeddingModel::AllMiniLML6V2Q, false),
            "all-MiniLM-L6-v2" => (EmbeddingModel::AllMiniLML6V2, false),
            _ => return Ok(None),
        };
        let te = TextEmbedding::try_new(
            InitOptions::new(em).with_cache_dir(crate::paths::model_cache_dir()),
        )?;
        Ok(Some(Self {
            model: te,
            tag: model.to_string(),
            bge,
        }))
    }
}

impl Embedder for FastEmbedder {
    fn id(&self) -> String {
        self.tag.clone()
    }

    fn embed(&self, texts: &[String], kind: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
        let prepped: Vec<String> = if self.bge && kind == EmbedKind::Query {
            texts
                .iter()
                .map(|t| format!("{BGE_QUERY_PREFIX}{t}"))
                .collect()
        } else {
            texts.to_vec()
        };
        self.model.embed(prepped, None)
    }

    // Tuned by sweeping the anthropic/skills corpus (scoped + global) against the
    // live installed skill set. bge is anisotropic: unrelated prompts still cosine
    // ~0.50-0.62 and genuine matches sit ~0.66+, so the floor is set at the knee
    // (0.64) — it rejects the noise tail while keeping real hits, trading one
    // borderline positive for two fewer false injections. The lone residual leak
    // is genuinely on-topic (a git skill on a git prompt). Margin 0.12 keeps only
    // near-peers of the leader. MiniLM shares this family tuning until it gets its
    // own corpus pass; it is an opt-in lite alternative.
    fn min_similarity(&self) -> f32 {
        0.64
    }

    fn score_margin(&self) -> f32 {
        0.12
    }
}
