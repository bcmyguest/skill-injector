//! Canonical on-disk locations, shared by every subcommand.
//!
//! - Index (the big embedding cache) lives under `$XDG_DATA_HOME/ski`.
//! - Session state (per-conversation dedup) lives under `$XDG_STATE_HOME/ski`.
//!
//! Both fall back to the XDG defaults relative to `$HOME` when the env vars are
//! unset, matching the rest of the toolchain.

use crate::hook::Host;
use std::path::PathBuf;

fn home() -> PathBuf {
    PathBuf::from(std::env::var_os("HOME").unwrap_or_default())
}

/// `$XDG_DATA_HOME/ski` (default `~/.local/share/ski`).
pub fn data_dir() -> PathBuf {
    std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local/share"))
        .join("ski")
}

/// Persistent skill index for `host`. Each host indexes only its own skill
/// library (see [`crate::config::Config::for_host`]), so the files are kept
/// apart; Claude keeps the original `index.json` to avoid orphaning it.
pub fn index_path(host: Host) -> PathBuf {
    let name = match host {
        Host::Claude => "index.json",
        Host::Opencode => "index-opencode.json",
    };
    data_dir().join(name)
}

/// `$XDG_STATE_HOME/ski` (default `~/.local/state/ski`).
pub fn state_dir() -> PathBuf {
    std::env::var_os("XDG_STATE_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| home().join(".local/state"))
        .join("ski")
}

/// Directory holding one JSON file per conversation.
pub fn sessions_dir() -> PathBuf {
    state_dir().join("sessions")
}

/// State file for a single conversation. The id is sanitized so a hostile or
/// odd session id can't escape the sessions directory.
pub fn session_path(session_id: &str) -> PathBuf {
    sessions_dir().join(format!("{}.json", sanitize(session_id)))
}

fn sanitize(id: &str) -> String {
    let s: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if s.is_empty() {
        "default".to_string()
    } else {
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_path_sanitizes_traversal() {
        let p = session_path("../../etc/passwd");
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(!name.contains('/'));
        assert!(!name.contains('.') || name.ends_with(".json"));
        assert_eq!(p.parent().unwrap(), sessions_dir());
    }

    #[test]
    fn empty_id_falls_back() {
        assert!(session_path("").ends_with("default.json"));
    }
}
