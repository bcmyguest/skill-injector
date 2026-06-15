//! `ski session-start` — the `SessionStart` path. Two jobs, both best-effort:
//!
//! 1. **Incremental reindex** so a session always sees newly added or edited
//!    skills (reuses unchanged embeddings; only the delta is re-embedded).
//! 2. **Re-arm on compaction**: when the session restarts from a compacted
//!    summary (`source == "compact"`), forget what was loaded so the relevant
//!    skills inject again into the fresh context.
//!
//! **Fail open**: any error is swallowed; never blocks session start.

use crate::config::Config;
use crate::hook::Host;
use crate::index::{self, Index};
use crate::session::Session;
use crate::{embed, paths, skill};
use serde::Deserialize;
use std::io::Read;

/// `SessionStart` payload: which conversation, and why it started
/// (`startup` | `resume` | `compact`).
#[derive(Debug, Default, Deserialize)]
struct RawEvent {
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    source: String,
}

/// Run for `host`. `host` scopes the reindex to that host's skill library and
/// its own index file (see [`crate::config::Config::for_host`]); the session
/// re-arm is host-agnostic (session ids are unique across hosts).
pub fn run(host: Host) -> anyhow::Result<()> {
    let _ = session_start(host); // fail open: never surface an error to the harness.
    Ok(())
}

fn session_start(host: Host) -> anyhow::Result<()> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let ev: RawEvent = serde_json::from_str(&buf).unwrap_or_default();

    reindex(host);

    if should_rearm(&ev.source) && !ev.session_id.is_empty() {
        let path = paths::session_path(&ev.session_id);
        let mut session = Session::load(&path);
        session.clear();
        let _ = session.save(&path);
    }
    Ok(())
}

/// Incrementally refresh the persisted index. Best-effort: any failure (no
/// skills, embedder build, IO) leaves the previous index untouched.
fn reindex(host: Host) {
    let cfg = Config::for_host(host);
    let Ok(skills) = skill::discover(&cfg.roots) else {
        return;
    };
    let Ok(embedder) = embed::build(&cfg.model) else {
        return;
    };
    let index_path = paths::index_path(host);
    let prev = Index::load(&index_path).ok().flatten();
    if let Ok(idx) = index::build(&skills, embedder.as_ref(), prev.as_ref()) {
        let _ = idx.save(&index_path);
    }
}

/// Only a compaction re-arms the session; `startup`/`resume` keep their ledger.
fn should_rearm(source: &str) -> bool {
    source.eq_ignore_ascii_case("compact")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::Source;

    #[test]
    fn only_compact_rearms() {
        assert!(should_rearm("compact"));
        assert!(!should_rearm("startup"));
        assert!(!should_rearm("resume"));
        assert!(!should_rearm(""));
    }

    #[test]
    fn clear_on_compact_empties_the_ledger() {
        let mut s = Session::default();
        s.mark("pdf", Source::Ski);
        s.mark("xlsx", Source::Model);
        if should_rearm("compact") {
            s.clear();
        }
        assert!(s.loaded.is_empty());
    }
}
