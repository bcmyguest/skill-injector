//! Per-conversation state: which skills are already in context, so we never
//! inject the same skill twice in one session.
//!
//! A skill is "loaded" either because **we** injected it ([`Source::Ski`]) or
//! because the **model** pulled it itself ([`Source::Model`], recorded by
//! `ski observe` in a later milestone). Dedup treats both the same — presence is
//! what matters; the source is kept for diagnostics and so a model-confirmed
//! load is never downgraded.
//!
//! All reads fail open: a missing or corrupt state file yields an empty session
//! rather than an error, so the hot path can never be blocked by bad state.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// We injected this skill.
    Ski,
    /// The model loaded this skill on its own.
    Model,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Session {
    /// skill id -> who put it in context.
    #[serde(default)]
    pub loaded: BTreeMap<String, Source>,
    /// Unix seconds of the last write (diagnostics only).
    #[serde(default)]
    pub updated: u64,
}

impl Session {
    /// Load state for a session, or an empty session if the file is missing or
    /// unreadable. Never errors.
    pub fn load(path: &Path) -> Session {
        fs::read_to_string(path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist state, stamping `updated`. Best-effort; callers in the hot path
    /// should ignore the result so state IO can't block a prompt.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut snapshot = self.clone();
        snapshot.updated = now_secs();
        fs::write(path, serde_json::to_string_pretty(&snapshot)?)?;
        Ok(())
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.contains_key(id)
    }

    /// Record that `id` is in context. A `Model` load always wins (it's a
    /// stronger signal); a `Ski` mark never overwrites an existing entry.
    pub fn mark(&mut self, id: &str, source: Source) {
        match source {
            Source::Model => {
                self.loaded.insert(id.to_string(), Source::Model);
            }
            Source::Ski => {
                self.loaded.entry(id.to_string()).or_insert(Source::Ski);
            }
        }
    }

    /// Forget everything — used to re-arm on compaction so skills can be
    /// re-injected into the fresh summary.
    pub fn clear(&mut self) {
        self.loaded.clear();
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mark_and_dedup() {
        let mut s = Session::default();
        assert!(!s.is_loaded("a"));
        s.mark("a", Source::Ski);
        assert!(s.is_loaded("a"));
    }

    #[test]
    fn model_load_is_not_downgraded() {
        let mut s = Session::default();
        s.mark("a", Source::Model);
        s.mark("a", Source::Ski); // later self-inject must not overwrite
        assert_eq!(s.loaded["a"], Source::Model);
    }

    #[test]
    fn ski_then_model_upgrades() {
        let mut s = Session::default();
        s.mark("a", Source::Ski);
        s.mark("a", Source::Model);
        assert_eq!(s.loaded["a"], Source::Model);
    }

    #[test]
    fn clear_re_arms() {
        let mut s = Session::default();
        s.mark("a", Source::Ski);
        s.clear();
        assert!(!s.is_loaded("a"));
    }

    #[test]
    fn source_serializes_lowercase() {
        let json = serde_json::to_string(&Source::Ski).unwrap();
        assert_eq!(json, "\"ski\"");
        let json = serde_json::to_string(&Source::Model).unwrap();
        assert_eq!(json, "\"model\"");
    }

    #[test]
    fn missing_file_is_empty_session() {
        let s = Session::load(Path::new("/nonexistent/ski/session.json"));
        assert!(s.loaded.is_empty());
    }

    #[test]
    fn roundtrip_through_json() {
        let mut s = Session::default();
        s.mark("git-attribution", Source::Ski);
        s.mark("uv-setup", Source::Model);
        let text = serde_json::to_string(&s).unwrap();
        let back: Session = serde_json::from_str(&text).unwrap();
        assert_eq!(back.loaded["git-attribution"], Source::Ski);
        assert_eq!(back.loaded["uv-setup"], Source::Model);
    }
}
