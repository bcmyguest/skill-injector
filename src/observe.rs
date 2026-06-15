//! `ski observe` — the `PostToolUse` path. Records skills the *model* pulled in
//! on its own (by reading a `SKILL.md`, or invoking the `Skill` tool) so the
//! hook's dedup never re-injects them.
//!
//! **Fail open is the contract**, same as the hook: any error — bad stdin,
//! missing index, IO failure — produces no output and exit 0. `observe` only
//! ever writes session state; it never emits to stdout.

use crate::hook::Host;
use crate::index::Index;
use crate::paths;
use crate::session::{Session, Source};
use serde::Deserialize;
use std::io::Read;
use std::path::Path;

/// `PostToolUse` payload. Claude sends `tool_name` + `tool_input`; the opencode
/// adapter is normalized to the same shape.
#[derive(Debug, Default, Deserialize)]
struct RawEvent {
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    tool_name: String,
    #[serde(default)]
    tool_input: ToolInput,
}

#[derive(Debug, Default, Deserialize)]
struct ToolInput {
    /// `Read` tool: the file the model opened.
    #[serde(default)]
    file_path: String,
    /// `Skill` tool: the invoked skill, plugin-namespaced (e.g.
    /// `document-skills:webapp-testing`). `normalize_skill_name` strips the
    /// prefix to the bare id we index by.
    #[serde(default)]
    skill: String,
}

/// Run the observer for `host`. `host` selects which per-host index resolves a
/// `Read` of a `SKILL.md` path back to its skill id (see
/// [`crate::paths::index_path`]); the `Skill`-tool path needs no index.
pub fn run(host: Host) -> anyhow::Result<()> {
    let _ = observe(host); // fail open: never surface an error to the harness.
    Ok(())
}

fn observe(host: Host) -> anyhow::Result<()> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let ev: RawEvent = serde_json::from_str(&buf).unwrap_or_default();
    if ev.session_id.is_empty() {
        return Ok(());
    }

    let idx = Index::load(&paths::index_path(host)).ok().flatten();
    let Some(id) = skill_id_for(idx.as_ref(), &ev.tool_name, &ev.tool_input) else {
        return Ok(());
    };

    let path = paths::session_path(&ev.session_id);
    let mut session = Session::load(&path);
    session.mark(&id, Source::Model);
    let _ = session.save(&path); // best-effort: state IO never blocks.
    Ok(())
}

/// Resolve the skill id a tool call loaded, if any. `Read` maps a `SKILL.md`
/// path through the index; `Skill` reads the invoked name directly.
fn skill_id_for(idx: Option<&Index>, tool: &str, input: &ToolInput) -> Option<String> {
    if tool.eq_ignore_ascii_case("Read") {
        let p = input.file_path.trim();
        if !is_skill_md(p) {
            return None;
        }
        return idx?.by_path(Path::new(p)).map(|e| e.id.clone());
    }
    if tool.eq_ignore_ascii_case("Skill") {
        let raw = input.skill.trim();
        if raw.is_empty() {
            return None;
        }
        return Some(normalize_skill_name(raw));
    }
    None
}

/// True when `path`'s final component is exactly `SKILL.md`.
fn is_skill_md(path: &str) -> bool {
    Path::new(path).file_name().and_then(|n| n.to_str()) == Some("SKILL.md")
}

/// Drop a `plugin:` namespace prefix so a tool-reported `document-skills:pdf`
/// matches the bare `pdf` id in our index and session ledger.
fn normalize_skill_name(raw: &str) -> String {
    raw.rsplit(':').next().unwrap_or(raw).trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Entry;

    fn idx_with(path: &str, id: &str) -> Index {
        Index {
            model: "m".into(),
            dim: 0,
            skills: vec![Entry {
                id: id.to_string(),
                name: id.to_string(),
                description: String::new(),
                path: path.to_string(),
                keywords: Vec::new(),
                hash: String::new(),
                embedding: Vec::new(),
            }],
        }
    }

    fn read_input(file_path: &str) -> ToolInput {
        ToolInput {
            file_path: file_path.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn read_of_skill_md_maps_to_id() {
        let idx = idx_with("/p/pdf/SKILL.md", "pdf");
        let got = skill_id_for(Some(&idx), "Read", &read_input("/p/pdf/SKILL.md"));
        assert_eq!(got.as_deref(), Some("pdf"));
    }

    #[test]
    fn read_of_other_file_is_ignored() {
        let idx = idx_with("/p/pdf/SKILL.md", "pdf");
        assert!(skill_id_for(Some(&idx), "Read", &read_input("/p/pdf/main.rs")).is_none());
    }

    #[test]
    fn read_of_unknown_skill_md_is_none() {
        let idx = idx_with("/p/pdf/SKILL.md", "pdf");
        assert!(skill_id_for(Some(&idx), "Read", &read_input("/p/other/SKILL.md")).is_none());
    }

    #[test]
    fn skill_tool_strips_namespace() {
        let input = ToolInput {
            skill: "document-skills:pdf".to_string(),
            ..Default::default()
        };
        assert_eq!(skill_id_for(None, "Skill", &input).as_deref(), Some("pdf"));
    }

    #[test]
    fn unrelated_tool_is_none() {
        assert!(skill_id_for(None, "Bash", &read_input("/p/pdf/SKILL.md")).is_none());
    }

    #[test]
    fn is_skill_md_only_matches_the_file() {
        assert!(is_skill_md("/a/b/SKILL.md"));
        assert!(!is_skill_md("/a/b/skill.md"));
        assert!(!is_skill_md("/a/SKILL.md.bak"));
    }
}
