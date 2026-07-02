//! The skill index: skill metadata plus the description embedding, persisted to
//! disk and reused incrementally (re-embed only entries whose content hash or
//! the embedding model changed).

use crate::embed::{EmbedKind, Embedder};
use crate::skill::Skill;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub path: String,
    pub keywords: Vec<String>,
    /// Trigger phrases for the phrase channel (see [`crate::skill::extract_phrases`]).
    /// `#[serde(default)]` so indexes written before this field still load.
    #[serde(default)]
    pub trigger_phrases: Vec<String>,
    pub hash: String,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Index {
    pub model: String,
    pub dim: usize,
    pub skills: Vec<Entry>,
}

impl Index {
    pub fn get(&self, id: &str) -> Option<&Entry> {
        self.skills.iter().find(|e| e.id == id)
    }

    /// Find the skill whose `SKILL.md` lives at `path`. Used by `ski observe` to
    /// map a file the model just read back to a skill id. Matches on the raw
    /// stored string first (cheap, and the common case), then falls back to
    /// canonicalized comparison so `./x` and `/abs/x` resolve to the same entry.
    pub fn by_path(&self, path: &Path) -> Option<&Entry> {
        let raw = path.to_string_lossy();
        if let Some(e) = self.skills.iter().find(|e| e.path == raw) {
            return Some(e);
        }
        let want = fs::canonicalize(path).ok()?;
        self.skills
            .iter()
            .find(|e| fs::canonicalize(&e.path).ok().as_deref() == Some(want.as_path()))
    }

    pub fn load(path: &Path) -> anyhow::Result<Option<Index>> {
        if !path.exists() {
            return Ok(None);
        }
        let data = fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&data)?))
    }

    /// Persist the index. Writes a per-process temp file then atomically renames
    /// it over the target, so a concurrent reader (a hook firing while
    /// `session-start`/`why` refreshes the index) never observes a half-written
    /// file — a torn read costs that hook a full re-embed of the library.
    /// Mirrors [`crate::session::Session::save`].
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
        fs::write(&tmp, json)?;
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }
}

/// Build (or incrementally refresh) the index for `skills` using `embedder`.
/// Entries in `prev` with a matching id+hash and the same model are reused; the
/// rest are embedded in one batch.
pub fn build(
    skills: &[Skill],
    embedder: &dyn Embedder,
    prev: Option<&Index>,
) -> anyhow::Result<Index> {
    let model = embedder.id();
    let mut entries: Vec<Option<Entry>> = vec![None; skills.len()];
    let mut to_embed: Vec<usize> = Vec::new();

    for (i, s) in skills.iter().enumerate() {
        let reuse = prev
            .filter(|p| p.model == model)
            .and_then(|p| p.get(&s.id))
            .filter(|e| e.hash == s.hash)
            .cloned();
        match reuse {
            // Reuse the cached embedding, but refresh the cheap content-derived
            // metadata (keywords, trigger phrases): an index written before these
            // were extracted has a matching hash, so without this the phrase
            // channel would stay dark until each skill's content next changed.
            Some(mut e) => {
                e.keywords = s.keywords.clone();
                e.trigger_phrases = s.trigger_phrases.clone();
                entries[i] = Some(e);
            }
            None => to_embed.push(i),
        }
    }

    if !to_embed.is_empty() {
        let texts: Vec<String> = to_embed
            .iter()
            .map(|&i| skills[i].description.clone())
            .collect();
        let embs = embedder.embed(&texts, EmbedKind::Document)?;
        for (k, &i) in to_embed.iter().enumerate() {
            let s = &skills[i];
            entries[i] = Some(Entry {
                id: s.id.clone(),
                name: s.name.clone(),
                description: s.description.clone(),
                path: s.path.display().to_string(),
                keywords: s.keywords.clone(),
                trigger_phrases: s.trigger_phrases.clone(),
                hash: s.hash.clone(),
                embedding: embs[k].clone(),
            });
        }
    }

    let skills: Vec<Entry> = entries.into_iter().flatten().collect();
    let dim = skills.first().map(|e| e.embedding.len()).unwrap_or(0);
    Ok(Index { model, dim, skills })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skill::Skill;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Embedder that counts how many texts it was asked to embed, to prove the
    /// incremental path reuses cached vectors instead of re-embedding.
    struct CountingEmbedder(AtomicUsize);
    impl Embedder for CountingEmbedder {
        fn id(&self) -> String {
            "counting".into()
        }
        fn embed(&self, texts: &[String], _: EmbedKind) -> anyhow::Result<Vec<Vec<f32>>> {
            self.0.fetch_add(texts.len(), Ordering::SeqCst);
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    fn skill(id: &str, hash: &str) -> Skill {
        Skill {
            id: id.to_string(),
            name: id.to_string(),
            description: format!("does {id}"),
            body_head: String::new(),
            keywords: Vec::new(),
            trigger_phrases: Vec::new(),
            path: std::path::PathBuf::from(format!("/s/{id}/SKILL.md")),
            hash: hash.to_string(),
        }
    }

    #[test]
    fn rebuild_with_prev_reuses_unchanged_embeddings() {
        let skills = vec![skill("a", "h1"), skill("b", "h2")];
        let e = CountingEmbedder(AtomicUsize::new(0));
        let first = build(&skills, &e, None).unwrap();
        assert_eq!(e.0.load(Ordering::SeqCst), 2); // both embedded

        // Same skills, prev supplied: nothing re-embeds (the `ski why` /
        // session-start hot path).
        let again = build(&skills, &e, Some(&first)).unwrap();
        assert_eq!(
            e.0.load(Ordering::SeqCst),
            2,
            "unchanged skills re-embedded"
        );
        assert_eq!(again.skills.len(), 2);

        // One skill's content changes: only that one re-embeds.
        let changed = vec![skill("a", "h1-new"), skill("b", "h2")];
        let _ = build(&changed, &e, Some(&first)).unwrap();
        assert_eq!(
            e.0.load(Ordering::SeqCst),
            3,
            "expected exactly one re-embed"
        );
    }

    #[test]
    fn save_is_atomic_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!("ski-index-save-{}", std::process::id()));
        let path = dir.join("index.json");
        let idx = Index {
            model: "m".into(),
            dim: 2,
            skills: vec![entry("a", "/s/a/SKILL.md")],
        };
        idx.save(&path).unwrap();
        let back = Index::load(&path).unwrap().unwrap();
        assert_eq!(back.skills[0].id, "a");
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| n != "index.json")
            .collect();
        assert!(leftovers.is_empty(), "temp file left behind: {leftovers:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    fn entry(id: &str, path: &str) -> Entry {
        Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            path: path.to_string(),
            keywords: Vec::new(),
            trigger_phrases: Vec::new(),
            hash: String::new(),
            embedding: Vec::new(),
        }
    }

    #[test]
    fn by_path_matches_stored_string() {
        let idx = Index {
            model: "m".into(),
            dim: 0,
            skills: vec![
                entry("pdf", "/skills/pdf/SKILL.md"),
                entry("xlsx", "/skills/xlsx/SKILL.md"),
            ],
        };
        assert_eq!(
            idx.by_path(Path::new("/skills/xlsx/SKILL.md")).unwrap().id,
            "xlsx"
        );
        assert!(idx.by_path(Path::new("/skills/none/SKILL.md")).is_none());
    }
}
