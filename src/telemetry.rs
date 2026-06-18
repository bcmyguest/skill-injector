//! Opt-in JSONL event log for debugging and calibration. **Disabled by default**
//! — every entry point short-circuits unless telemetry is enabled, via either
//! `telemetry = true` in `~/.config/ski/config.toml` or a truthy `SKI_TELEMETRY`
//! env var (`1|true|yes|on`, e.g. in the `env` block of `~/.claude/settings.json`).
//! Each entry point calls [`init`] right after `Config::load` to reflect the
//! config flag into the process; [`enabled`] is then config-OR-env.
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

/// Set from the loaded `Config` (`telemetry = true` in `config.toml`) by each
/// entry point before it records anything. Lets the config file enable telemetry
/// without an env var; the env var still works on its own for the hook's `env`
/// block. Defaults to off until [`init`] runs.
static CONFIG_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Reflect `cfg.telemetry` into the process so [`enabled`] sees it. Call once,
/// right after `Config::load`, in any entry point that may record events.
pub fn init(config_enabled: bool) {
    CONFIG_ENABLED.store(config_enabled, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the event log is active. Cheap; called at every entry point so a
/// disabled log costs one env lookup and nothing else. On when the config flag
/// (via [`init`]) *or* a truthy `SKI_TELEMETRY` env var is set.
pub fn enabled() -> bool {
    CONFIG_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
        || matches!(
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
/// `Skill` tool) or `"read"` (opened the `SKILL.md`). `prompt` is the active
/// prompt the hook stashed in session state (empty if none), letting `ski
/// history` tie a recall miss back to the call that triggered it.
pub fn record_use(session_id: &str, skill_id: &str, via: &str, prompt: &str) {
    if !enabled() {
        return;
    }
    let mut ev = json!({
        "ts": now_ms(),
        "kind": "use",
        "session": session_id,
        "skill": skill_id,
        "via": via,
    });
    if !prompt.is_empty() {
        ev["prompt"] = json!(prompt);
    }
    append(&ev);
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
        record_use("s", "pdf", "skill", "");
    }

    #[test]
    fn stage_strings() {
        assert_eq!(stage_str(Stage::Cosine), "cosine");
        assert_eq!(stage_str(Stage::Rerank), "rerank");
    }
}
