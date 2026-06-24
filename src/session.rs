//! Per-conversation state: which skills are already in context, and at what
//! confidence we last recommended them, so dedup can be *score-aware* rather than
//! "seen once, suppressed forever".
//!
//! A skill is "loaded" either because **we** recommended it ([`Source::Ski`],
//! with the confidence we showed) or because the **model** pulled it itself
//! ([`Source::Model`], recorded by `ski observe`). The two are treated
//! differently by [`Session::should_recommend`]:
//! - **used** (`Model`) — never recommend again.
//! - **recommended, unused** (`Ski`) — re-recommend only once it newly reaches
//!   HIGH confidence (we get one stronger nudge; after a HIGH showing, never).
//!
//! All reads fail open: a missing or corrupt state file yields an empty session
//! rather than an error, so the hot path can never be blocked by bad state.

use serde::de::Deserializer;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// We recommended this skill.
    Ski,
    /// The model loaded this skill on its own.
    Model,
}

/// What we know about a skill already in context: who put it there, and (for a
/// `Ski` recommendation) the confidence we displayed. `Model` loads carry the
/// last confidence we'd shown, or `0.0` if we never recommended it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct Record {
    pub source: Source,
    pub confidence: f32,
}

// Backward-compatible read: an older state file stored each value as a bare
// `"ski"`/`"model"` string. Accept either that (confidence 0) or the current
// `{source, confidence}` object, so an in-flight session survives an upgrade.
impl<'de> Deserialize<'de> for Record {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Repr {
            Bare(Source),
            Full {
                source: Source,
                #[serde(default)]
                confidence: f32,
            },
        }
        Ok(match Repr::deserialize(d)? {
            Repr::Bare(source) => Record {
                source,
                confidence: 0.0,
            },
            Repr::Full { source, confidence } => Record { source, confidence },
        })
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Session {
    /// skill id -> how it got into context (and at what confidence).
    #[serde(default)]
    pub loaded: BTreeMap<String, Record>,
    /// The most recent user prompt in this conversation. Stashed by the hook
    /// **only when telemetry is on**, so a later self-load seen by `ski observe`
    /// (a recall miss — the model loaded a skill we never recommended) can be
    /// tied back to the prompt that was active. Empty otherwise; never serialized
    /// when empty, so the non-telemetry hot path leaves the file unchanged.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub last_prompt: String,
    /// Recent user prompts in this conversation, oldest-first, bounded. Drives
    /// query-side context enrichment: a vague follow-up ("now do the other one")
    /// is disambiguated by the turns that preceded it. Maintained only when the
    /// context feature is enabled (`Config::context_depth > 0`), so the default
    /// hot path neither writes nor carries it. `#[serde(default)]` + skip-when-empty
    /// keeps it invisible to indexes/sessions written before it existed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent_prompts: Vec<String>,
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
    ///
    /// Writes a per-process temp file then atomically renames it over the target,
    /// so a concurrent reader (another hook/observe process sharing the
    /// `session_id`) never observes a half-written file — a torn read used to
    /// silently reset the session and re-arm dedup. The lost-update window — two
    /// writers racing the load→mutate→save and one dropping the other's mark —
    /// remains; it costs at most a missed dedup (a re-injection), never
    /// corruption, and closing it would need an advisory lock.
    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut snapshot = self.clone();
        snapshot.updated = now_secs();
        let json = serde_json::to_string_pretty(&snapshot)?;
        let tmp = path.with_extension(format!("tmp.{}.{}", std::process::id(), now_nanos()));
        fs::write(&tmp, json)?;
        if let Err(e) = fs::rename(&tmp, path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        Ok(())
    }

    pub fn is_loaded(&self, id: &str) -> bool {
        self.loaded.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&Record> {
        self.loaded.get(id)
    }

    /// Whether `id` should be recommended now, at `new_conf`, given what we
    /// already know. The two dedup rules:
    /// - a **used** skill (`Source::Model`) is never recommended again;
    /// - a **recommended-but-unused** skill (`Source::Ski`) is re-recommended
    ///   only when it newly reaches `high` confidence (it was shown below `high`
    ///   before — a clearer prompt earns one stronger nudge; after a HIGH
    ///   showing, never).
    pub fn should_recommend(&self, id: &str, new_conf: f32, high: f32) -> bool {
        match self.loaded.get(id) {
            None => true,
            Some(r) if r.source == Source::Model => false,
            Some(r) => new_conf >= high && r.confidence < high,
        }
    }

    /// Record that we recommended `id` at `confidence`. Stores the confidence we
    /// just showed (so the next-turn `should_recommend` test is accurate), but
    /// never downgrades a `Model` load — once the model used a skill it stays
    /// used.
    pub fn mark_recommended(&mut self, id: &str, confidence: f32) {
        match self.loaded.get(id) {
            Some(r) if r.source == Source::Model => {}
            _ => {
                self.loaded.insert(
                    id.to_string(),
                    Record {
                        source: Source::Ski,
                        confidence,
                    },
                );
            }
        }
    }

    /// Record that the model loaded `id` itself. Always wins (the strongest
    /// signal); keeps any confidence we'd previously shown for diagnostics.
    pub fn mark_used(&mut self, id: &str) {
        let confidence = self.loaded.get(id).map(|r| r.confidence).unwrap_or(0.0);
        self.loaded.insert(
            id.to_string(),
            Record {
                source: Source::Model,
                confidence,
            },
        );
    }

    /// Generic mark, kept for callers/tests that don't carry a confidence:
    /// `Model` via [`mark_used`], `Ski` as a confidence-0 first sighting that
    /// never overwrites an existing entry.
    pub fn mark(&mut self, id: &str, source: Source) {
        match source {
            Source::Model => self.mark_used(id),
            Source::Ski => {
                self.loaded.entry(id.to_string()).or_insert(Record {
                    source: Source::Ski,
                    confidence: 0.0,
                });
            }
        }
    }

    /// Append `prompt` to the rolling context window, keeping at most `max` of the
    /// most recent prompts (oldest dropped first). A blank prompt, or one identical
    /// to the immediately previous entry (a resubmit), is ignored so the window
    /// holds distinct conversational turns. `max == 0` disables the window entirely
    /// (the feature-off path).
    pub fn push_prompt(&mut self, prompt: &str, max: usize) {
        let p = prompt.trim();
        if max == 0 || p.is_empty() {
            return;
        }
        if self.recent_prompts.last().map(String::as_str) == Some(p) {
            return;
        }
        self.recent_prompts.push(p.to_string());
        let len = self.recent_prompts.len();
        if len > max {
            self.recent_prompts.drain(0..len - max);
        }
    }

    /// Forget everything — used to re-arm on compaction so skills can be
    /// re-injected into the fresh summary.
    pub fn clear(&mut self) {
        self.loaded.clear();
        self.recent_prompts.clear();
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Nanosecond stamp, used only to make the atomic-write temp path unique per
/// writer so two concurrent saves can't collide on the same temp file.
fn now_nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
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
        assert_eq!(s.loaded["a"].source, Source::Model);
    }

    #[test]
    fn ski_then_model_upgrades() {
        let mut s = Session::default();
        s.mark("a", Source::Ski);
        s.mark("a", Source::Model);
        assert_eq!(s.loaded["a"].source, Source::Model);
    }

    #[test]
    fn used_skill_is_never_recommended() {
        let mut s = Session::default();
        s.mark_used("a");
        // Even a maxed-out confidence can't resurrect a used skill.
        assert!(!s.should_recommend("a", 1.0, 0.80));
    }

    #[test]
    fn unseen_skill_is_recommended() {
        let s = Session::default();
        assert!(s.should_recommend("a", 0.40, 0.80)); // any confidence, never seen
    }

    #[test]
    fn repeat_only_on_rise_into_high() {
        let mut s = Session::default();
        s.mark_recommended("a", 0.60); // shown at medium
        assert!(!s.should_recommend("a", 0.70, 0.80)); // still below high -> no repeat
        assert!(s.should_recommend("a", 0.90, 0.80)); // newly high -> one nudge
    }

    #[test]
    fn no_repeat_after_high_showing() {
        let mut s = Session::default();
        s.mark_recommended("a", 0.90); // already shown at high
        assert!(!s.should_recommend("a", 0.95, 0.80)); // even higher -> still suppressed
    }

    #[test]
    fn mark_recommended_does_not_downgrade_model() {
        let mut s = Session::default();
        s.mark_used("a");
        s.mark_recommended("a", 0.99);
        assert_eq!(s.loaded["a"].source, Source::Model);
    }

    #[test]
    fn legacy_bare_string_value_still_loads() {
        // Pre-confidence on-disk format: value is a bare source string.
        let json = r#"{"loaded":{"a":"ski","b":"model"},"updated":0}"#;
        let s: Session = serde_json::from_str(json).unwrap();
        assert_eq!(s.loaded["a"].source, Source::Ski);
        assert_eq!(s.loaded["a"].confidence, 0.0);
        assert_eq!(s.loaded["b"].source, Source::Model);
    }

    #[test]
    fn clear_re_arms() {
        let mut s = Session::default();
        s.mark("a", Source::Ski);
        s.push_prompt("set up pytest", 3);
        s.clear();
        assert!(!s.is_loaded("a"));
        assert!(s.recent_prompts.is_empty()); // window re-armed too
    }

    #[test]
    fn push_prompt_bounds_window_oldest_first() {
        let mut s = Session::default();
        for p in ["one", "two", "three", "four"] {
            s.push_prompt(p, 3);
        }
        // Capped at 3, oldest ("one") dropped, order preserved.
        assert_eq!(s.recent_prompts, ["two", "three", "four"]);
    }

    #[test]
    fn push_prompt_ignores_blank_and_consecutive_dupes() {
        let mut s = Session::default();
        s.push_prompt("  ", 3); // blank -> ignored
        s.push_prompt("set up pytest", 3);
        s.push_prompt("set up pytest", 3); // immediate resubmit -> ignored
        s.push_prompt("now the other one", 3);
        assert_eq!(s.recent_prompts, ["set up pytest", "now the other one"]);
    }

    #[test]
    fn push_prompt_zero_max_disables_window() {
        let mut s = Session::default();
        s.push_prompt("anything", 0);
        assert!(s.recent_prompts.is_empty()); // feature-off: never records
    }

    #[test]
    fn recent_prompts_absent_when_empty_in_json() {
        // skip_serializing_if keeps the field out of the on-disk form for the
        // default (feature-off) path, so existing readers/writers are unaffected.
        let s = Session::default();
        let json = serde_json::to_string(&s).unwrap();
        assert!(!json.contains("recent_prompts"), "got {json}");
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
    fn save_then_load_roundtrips_and_leaves_no_temp() {
        let dir = std::env::temp_dir().join(format!(
            "ski-session-save-{}-{}",
            std::process::id(),
            now_nanos()
        ));
        let path = dir.join("conv.json");
        let mut s = Session::default();
        s.mark("uv-setup", Source::Ski);
        s.save(&path).unwrap();

        let back = Session::load(&path);
        assert_eq!(back.loaded["uv-setup"].source, Source::Ski);
        // The temp file used by the atomic rename must not survive the write.
        let leftovers: Vec<_> = fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name())
            .filter(|n| n != "conv.json")
            .collect();
        assert!(leftovers.is_empty(), "temp file left behind: {leftovers:?}");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn roundtrip_through_json() {
        let mut s = Session::default();
        s.mark("git-attribution", Source::Ski);
        s.mark("uv-setup", Source::Model);
        let text = serde_json::to_string(&s).unwrap();
        let back: Session = serde_json::from_str(&text).unwrap();
        assert_eq!(back.loaded["git-attribution"].source, Source::Ski);
        assert_eq!(back.loaded["uv-setup"].source, Source::Model);
    }
}
