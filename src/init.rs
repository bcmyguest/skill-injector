//! `ski init` — one-shot setup of ski's hooks for a host into the user's config.
//!
//! For **opencode** it drops the bundled `ski.ts` plugin into the global plugin
//! directory. For **Claude Code** it merges the three hooks straight into
//! `~/.claude/settings.json` — the install path for users who can't (or don't
//! want to) go through the `/plugin` marketplace. The Claude merge is additive
//! and idempotent: it never rewrites unrelated settings and won't double-add a
//! hook that is already wired (whether by a previous `init` or by the plugin).

use crate::hook::Host;
use crate::paths;
use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::fs;

/// The bundled opencode plugin, embedded at build time so an installed `ski`
/// needs no access to the source tree to set opencode up.
const OPENCODE_PLUGIN: &str = include_str!("../opencode/ski.ts");

/// The Claude hooks ski installs, as `(event, matcher, ski subcommand)`. The
/// matchers mirror `hooks/hooks.json` so a manual install behaves identically to
/// the marketplace plugin.
const CLAUDE_HOOKS: &[(&str, Option<&str>, &str)] = &[
    ("UserPromptSubmit", None, "hook"),
    ("PostToolUse", Some("Read|Skill"), "observe"),
    (
        "SessionStart",
        Some("startup|resume|compact"),
        "session-start",
    ),
];

pub fn run(host: Host, global: bool) -> Result<()> {
    if !global {
        anyhow::bail!(
            "per-project install is not implemented yet; pass -g/--global for a \
             user-wide install"
        );
    }
    match host {
        Host::Opencode => init_opencode(),
        Host::Claude => init_claude(),
    }
}

/// Write the bundled plugin to `~/.config/opencode/plugin/ski.ts`. Overwriting is
/// safe — the file is ours and regenerable — and keeps an existing install up to
/// date with this binary's version.
fn init_opencode() -> Result<()> {
    let dir = paths::opencode_plugin_dir();
    fs::create_dir_all(&dir)
        .with_context(|| format!("creating opencode plugin dir {}", dir.display()))?;
    let dest = dir.join("ski.ts");
    fs::write(&dest, OPENCODE_PLUGIN).with_context(|| format!("writing {}", dest.display()))?;
    println!("installed opencode plugin -> {}", dest.display());
    print_next_steps("opencode");
    Ok(())
}

/// Merge ski's hooks into `~/.claude/settings.json`, creating the file if absent
/// and backing up any existing one to `settings.json.bak` first.
fn init_claude() -> Result<()> {
    let path = paths::claude_settings_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    }

    let mut root: Value = if path.exists() {
        let raw =
            fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        fs::write(path.with_extension("json.bak"), &raw)
            .with_context(|| format!("backing up {}", path.display()))?;
        serde_json::from_str(&raw)
            .with_context(|| format!("{} is not valid JSON", path.display()))?
    } else {
        json!({})
    };

    // The hook points at this very binary by absolute path, so it works no matter
    // what PATH the hook subprocess inherits.
    let exe = std::env::current_exe().context("locating the ski binary")?;
    let exe = exe.display();

    let obj = root
        .as_object_mut()
        .context("settings.json must be a JSON object")?;
    let hooks = obj
        .entry("hooks")
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .context("\"hooks\" in settings.json must be an object")?;

    let mut added = 0;
    for &(event, matcher, sub) in CLAUDE_HOOKS {
        let arr = hooks
            .entry(event)
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .with_context(|| format!("\"hooks.{event}\" must be an array"))?;
        if arr.iter().any(|g| group_runs_ski(g, sub)) {
            continue; // already wired (by a prior init or the plugin)
        }
        let command = format!("\"{exe}\" {sub} --host claude");
        let entry = match matcher {
            Some(m) => {
                json!({ "matcher": m, "hooks": [{ "type": "command", "command": command }] })
            }
            None => json!({ "hooks": [{ "type": "command", "command": command }] }),
        };
        arr.push(entry);
        added += 1;
    }

    let mut out = serde_json::to_string_pretty(&root)?;
    out.push('\n');
    fs::write(&path, out).with_context(|| format!("writing {}", path.display()))?;

    if added == 0 {
        println!("ski hooks already present in {}", path.display());
    } else {
        println!("wired {added} ski hook(s) into {}", path.display());
    }
    print_next_steps("claude");
    Ok(())
}

/// Post-install pointers. The two things every new install trips over: the
/// first ranked prompt otherwise blocks on the one-time model download, and a
/// zero-skill library silently injects nothing.
fn print_next_steps(host: &str) {
    println!("next steps:");
    println!(
        "  ski index --host {host}    # pre-download the embedding models (one-time, ~275 MB)\n\
         \x20                            and build the index — otherwise your first prompt blocks on it"
    );
    println!("  ski why \"set up a python project\"    # verify skills are discovered and ranked");
}

/// Whether a settings.json hook group already runs `ski <sub> --host claude` —
/// matches both the marketplace command (via `ski-bootstrap.sh`) and a direct
/// binary call, since both end in `<sub> --host claude`.
fn group_runs_ski(group: &Value, sub: &str) -> bool {
    let needle = format!("{sub} --host claude");
    group
        .get("hooks")
        .and_then(Value::as_array)
        .map(|hs| {
            hs.iter().any(|h| {
                h.get("command")
                    .and_then(Value::as_str)
                    .is_some_and(|c| c.contains(&needle))
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_existing_marketplace_hook() {
        let g = json!({
            "hooks": [{
                "type": "command",
                "command": "bash \"${CLAUDE_PLUGIN_ROOT}/scripts/ski-bootstrap.sh\" hook --host claude"
            }]
        });
        assert!(group_runs_ski(&g, "hook"));
        assert!(!group_runs_ski(&g, "observe"));
    }

    #[test]
    fn detects_direct_binary_hook() {
        let g = json!({
            "hooks": [{ "type": "command", "command": "\"/home/u/.local/bin/ski\" observe --host claude" }]
        });
        assert!(group_runs_ski(&g, "observe"));
    }

    #[test]
    fn ignores_unrelated_hook() {
        let g = json!({
            "hooks": [{ "type": "command", "command": "echo hi" }]
        });
        assert!(!group_runs_ski(&g, "hook"));
    }
}
