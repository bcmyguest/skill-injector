//! Opt-in JSONL event log for debugging and calibration. **Disabled by default**
//! — every entry point short-circuits unless telemetry is enabled, via either
//! `telemetry = true` in `~/.config/ski/config.toml` or a truthy `SKI_TELEMETRY`
//! env var (`1|true|yes|on`, e.g. in the `env` block of `~/.claude/settings.json`).
//! Each entry point calls [`init`] right after `Config::load` to reflect the
//! config flag into the process; [`enabled`] is then config-OR-env.
//!
//! Two event kinds, appended one JSON object per line to
//! `$XDG_STATE_HOME/ski/telemetry.jsonl`:
//! - `recommend` — what `ski hook` decided on a prompt. Emitted on **every**
//!   ranked prompt, including the ones where ski injects nothing: the prompt, the
//!   stage, the top-K `considered` ranking (id + raw stage score) the chooser
//!   produced *before* the gate, the `candidates` that cleared the gate, which
//!   ids survived the char budget (`injected`), and an `abstained` reason when
//!   nothing was injected. The always-present `considered` list is what lets a
//!   later analysis see where ski ranked a skill on a prompt it stayed silent on.
//! - `use` — a skill the model loaded itself (seen by `ski observe`) — i.e. the
//!   host's own (native) skill chooser's pick. Joining a `use` to the prompt's
//!   `recommend` event by `session` + `prompt` tells us whether the native pick
//!   was something ski injected, ranked-but-abstained-on, or never surfaced.
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

/// Record the hook's decision on a prompt. Emitted on every ranked prompt, even
/// when ski injects nothing, so the always-present `considered` ranking records
/// where ski placed each skill on a prompt it stayed silent on.
///
/// - `considered` — the top-K of the ranking the winning stage produced, *before*
///   the gate: `(id, raw stage score)` (cosine-blend for stage 1, reranker logit
///   for stage 2). This is the chooser's view; joining the native pick (a `use`
///   event) against it shows whether ski near-missed or never surfaced it.
/// - `recs` — the candidates that cleared the gate (empty on abstention).
/// - `injected` — the subset that fit the char budget (id + shown confidence).
/// - `abstained` — why nothing was injected (`Some("below_gate")` etc.), or
///   `None` when an injection was emitted.
pub fn record_recommend(
    session_id: &str,
    prompt: &str,
    stage: Stage,
    considered: &[(String, f32)],
    recs: &[Rec],
    injected: &[(String, f32)],
    abstained: Option<&str>,
) {
    if !enabled() {
        return;
    }
    let considered: Vec<_> = considered
        .iter()
        .map(|(id, s)| json!({ "id": id, "score": s }))
        .collect();
    let candidates: Vec<_> = recs
        .iter()
        .map(|r| json!({ "id": r.id, "confidence": r.confidence }))
        .collect();
    let injected: Vec<_> = injected
        .iter()
        .map(|(id, c)| json!({ "id": id, "confidence": c }))
        .collect();
    let mut ev = json!({
        "ts": now_ms(),
        "kind": "recommend",
        "session": session_id,
        "prompt": prompt,
        "stage": stage_str(stage),
        "considered": considered,
        "candidates": candidates,
        "injected": injected,
    });
    if let Some(reason) = abstained {
        ev["abstained"] = json!(reason);
    }
    append(&ev);
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
        Stage::Lexical => "lexical",
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
