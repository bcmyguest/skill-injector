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

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_string_pretty(self)?)?;
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
            Some(e) => entries[i] = Some(e),
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

    fn entry(id: &str, path: &str) -> Entry {
        Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            path: path.to_string(),
            keywords: Vec::new(),
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
