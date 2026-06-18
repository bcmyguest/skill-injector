//! Opt-in JSONL event log for debugging and calibration. **Disabled by default**
//! — every entry point short-circuits unless `SKI_TELEMETRY` is truthy
//! (`1|true|yes|on`), set in the hook's environment (e.g. the `env` block of
//! `~/.claude/settings.json`).
//!
//! Two event kinds, appended one JSON object per line to
//! `$XDG_STATE_HOME/ski/telemetry.jsonl`:
//! - `recommend` — what `ski hook` injected: the prompt, the stage, every
//!   candidate's confidence, and which ids actually survived the char budget.
//! - `use` — a skill the model loaded itself (seen by `ski observe`). Joining a
//!   `use` to an earlier `recommend` by `session` + `skill` tells us whether a
//!   recommendation was acted on.
//!
//! Best-effort, like the rest of the hot path: any IO/serialization failure is
//! swallowed so telemetry can never block or fail a prompt.

use crate::confidence::Stage;
use crate::inject::Rec;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::Write;
use std::time::{SystemTime, UNIX_EPOCH};

/// Whether the event log is active. Cheap; called at every entry point so a
/// disabled log costs one env lookup and nothing else.
pub fn enabled() -> bool {
    matches!(
        std::env::var("SKI_TELEMETRY").ok().as_deref(),
        Some("1") | Some("true") | Some("yes") | Some("on")
    )
}

/// Record an injection: the candidates that cleared the gate (`recs`) and the
/// subset that fit the budget (`injected`, id + shown confidence).
pub fn record_recommend(
    session_id: &str,
    prompt: &str,
    stage: Stage,
    recs: &[Rec],
    injected: &[(String, f32)],
) {
    if !enabled() {
        return;
    }
    let candidates: Vec<_> = recs
        .iter()
        .map(|r| json!({ "id": r.id, "confidence": r.confidence }))
        .collect();
    let injected: Vec<_> = injected
        .iter()
        .map(|(id, c)| json!({ "id": id, "confidence": c }))
        .collect();
    append(&json!({
        "ts": now_ms(),
        "kind": "recommend",
        "session": session_id,
        "prompt": prompt,
        "stage": stage_str(stage),
        "candidates": candidates,
        "injected": injected,
    }));
}

/// Record that the model loaded `skill_id` itself. `via` is `"skill"` (the
/// `Skill` tool) or `"read"` (opened the `SKILL.md`).
pub fn record_use(session_id: &str, skill_id: &str, via: &str) {
    if !enabled() {
        return;
    }
    append(&json!({
        "ts": now_ms(),
        "kind": "use",
        "session": session_id,
        "skill": skill_id,
        "via": via,
    }));
}

fn append(ev: &serde_json::Value) {
    let path = crate::paths::telemetry_path();
    let _ = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut f = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(f, "{ev}")?;
        Ok(())
    })();
}

fn stage_str(stage: Stage) -> &'static str {
    match stage {
        Stage::Cosine => "cosine",
        Stage::Rerank => "rerank",
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_by_default() {
        // The test process has no SKI_TELEMETRY set.
        std::env::remove_var("SKI_TELEMETRY");
        assert!(!enabled());
        // record_* must be no-ops when disabled (no panic, no file).
        record_use("s", "pdf", "skill");
    }

    #[test]
    fn stage_strings() {
        assert_eq!(stage_str(Stage::Cosine), "cosine");
        assert_eq!(stage_str(Stage::Rerank), "rerank");
    }
}
