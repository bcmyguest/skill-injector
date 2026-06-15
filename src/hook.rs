//! `ski hook` — the hot path. Reads a hook event on stdin, decides which skills
//! to inject, writes the host's injection contract on stdout.
//!
//! **Fail open is the contract.** Any error — bad stdin, missing index, IO
//! failure — results in an empty injection and exit 0, never a blocked prompt.
//!
//! Output by host:
//! - Claude (`UserPromptSubmit`): `{hookSpecificOutput:{hookEventName,
//!   additionalContext}}`, or nothing when there's no injection.
//! - opencode: `{skills:[...], inject:"..."}` always (the TS adapter parses it).

use crate::config::{Config, Strength};
use crate::embed::{self, EmbedKind};
use crate::index::{self, Index};
use crate::rank::Hit;
use crate::session::{Session, Source};
use crate::{inject, paths, rank, rerank, skill};
use serde::Deserialize;
use std::io::Read;
use std::str::FromStr;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Host {
    Claude,
    Opencode,
}

impl FromStr for Host {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "claude" => Ok(Host::Claude),
            "opencode" => Ok(Host::Opencode),
            other => anyhow::bail!("unknown host '{other}' (expected 'claude' or 'opencode')"),
        }
    }
}

/// The normalized hook event. Claude's `UserPromptSubmit` payload already uses
/// these field names; the opencode adapter sends the same shape.
#[derive(Debug, Default, Deserialize)]
struct RawEvent {
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    session_id: String,
    #[serde(default)]
    cwd: String,
}

#[derive(Debug, Default)]
struct Decision {
    inject: String,
    skills: Vec<String>,
}

/// Run the hook for `host`. Always exits 0 (fail open).
pub fn run(host: Host) -> anyhow::Result<()> {
    let decision = decide(host).unwrap_or_default();
    let out = match host {
        Host::Claude => render_claude(&decision),
        Host::Opencode => render_opencode(&decision),
    };
    if !out.is_empty() {
        println!("{out}");
    }
    Ok(())
}

fn decide(host: Host) -> anyhow::Result<Decision> {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf)?;
    let event: RawEvent = serde_json::from_str(&buf).unwrap_or_default();
    if event.prompt.trim().is_empty() {
        return Ok(Decision::default());
    }
    let _ = &event.cwd; // project-scoped config/roots arrive in a later milestone.

    let mut cfg = Config::for_host(host);
    let embedder = embed::build(&cfg.model)?;
    cfg.calibrate_to(embedder.as_ref());
    let idx = load_or_build_index(&cfg, embedder.as_ref(), host)?;
    if idx.skills.is_empty() {
        return Ok(Decision::default());
    }

    let query = embedder
        .embed(std::slice::from_ref(&event.prompt), EmbedKind::Query)?
        .remove(0);
    let hits = rank::rank_all(&query, &event.prompt, &idx, &cfg);

    let session_path = paths::session_path(&event.session_id);
    let mut session = Session::load(&session_path);
    // Stage 2: when stage-1 is ambiguous, let the cross-encoder decide; otherwise
    // (confident winner or nothing relevant) keep the cheap stage-1 result.
    let selected = match rerank::is_ambiguous(&hits, &cfg)
        .then(|| rerank::rerank(&hits, &idx, &event.prompt, &cfg))
        .flatten()
    {
        Some(reranked) => select_reranked(reranked, &cfg, &session),
        None => select(hits, &cfg, &session),
    };
    if selected.is_empty() {
        return Ok(Decision::default());
    }

    let strength = resolve_strength(cfg.directive_strength, host);
    let (text, ids) = inject::build(&selected, &idx, cfg.inject_mode, strength, cfg.char_budget);
    if text.is_empty() {
        return Ok(Decision::default());
    }

    for id in &ids {
        session.mark(id, Source::Ski);
    }
    let _ = session.save(&session_path); // best-effort: state IO never blocks.

    Ok(Decision {
        inject: text,
        skills: ids,
    })
}

/// Load the persisted index; build it on first run so the hook works before an
/// explicit `ski index`. Rebuilds when the stored index was made by a different
/// embedder — its vectors live in another space (and often another dimension), so
/// cosine against a query from the current embedder would be meaningless. This
/// makes switching embedders (e.g. bag-of-words -> bge) self-healing in the hot
/// path rather than only on the next `SessionStart`.
fn load_or_build_index(
    cfg: &Config,
    embedder: &dyn embed::Embedder,
    host: Host,
) -> anyhow::Result<Index> {
    let path = paths::index_path(host);
    if let Some(idx) = Index::load(&path)? {
        if idx.model == embedder.id() {
            return Ok(idx);
        }
    }
    let skills = skill::discover(&cfg.roots)?;
    let idx = index::build(&skills, embedder, None)?;
    let _ = idx.save(&path);
    Ok(idx)
}

/// Apply the guardrails: drop denied skills, keep those that clear both the
/// absolute floor (`min_similarity`) and the relative gate (within `score_margin`
/// of the best-scoring skill) — or are forced on a keyword hit — then drop any
/// already in context this session and cap at `max_skills`.
///
/// The relative gate is measured against the global best **before** session
/// dedup, so re-injecting a prompt whose strong matches are already loaded falls
/// silent instead of scraping the weak tail.
fn select(hits: Vec<Hit>, cfg: &Config, session: &Session) -> Vec<Hit> {
    let top = hits.first().map(|h| h.score).unwrap_or(0.0);
    hits.into_iter()
        .filter(|h| !cfg.deny.contains(&h.id))
        .filter(|h| {
            let forced = cfg.force.contains(&h.id) && h.keyword > 0.0;
            forced || (h.score >= cfg.min_similarity && h.score >= top - cfg.score_margin)
        })
        .filter(|h| !session.is_loaded(&h.id))
        .take(cfg.max_skills)
        .collect()
}

/// Guardrails for the reranked path. The reranker scores already passed their own
/// floor/margin in [`rerank::passes`], so this only drops denied and
/// already-loaded skills and caps the count — the reranker-scale equivalent of the
/// tail of [`select`].
fn select_reranked(reranked: Vec<Hit>, cfg: &Config, session: &Session) -> Vec<Hit> {
    rerank::passes(&reranked, cfg)
        .into_iter()
        .filter(|h| !cfg.deny.contains(&h.id))
        .filter(|h| !session.is_loaded(&h.id))
        .take(cfg.max_skills)
        .collect()
}

/// Resolve [`Strength::Auto`] from the host: Claude has a strong native chooser
/// (a nudge suffices); opencode's local models need an imperative.
fn resolve_strength(strength: Strength, host: Host) -> Strength {
    match strength {
        Strength::Auto => match host {
            Host::Claude => Strength::Soft,
            Host::Opencode => Strength::Hard,
        },
        other => other,
    }
}

fn render_claude(d: &Decision) -> String {
    if d.inject.is_empty() {
        return String::new();
    }
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": d.inject,
        }
    })
    .to_string()
}

fn render_opencode(d: &Decision) -> String {
    serde_json::json!({ "skills": d.skills, "inject": d.inject }).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    fn hit(id: &str, score: f32, keyword: f32) -> Hit {
        Hit {
            id: id.to_string(),
            name: id.to_string(),
            cosine: score - keyword,
            keyword,
            score,
        }
    }

    #[test]
    fn host_parse() {
        assert_eq!("claude".parse::<Host>().unwrap(), Host::Claude);
        assert_eq!("OpenCode".parse::<Host>().unwrap(), Host::Opencode);
        assert!("bogus".parse::<Host>().is_err());
    }

    #[test]
    fn raw_event_parses_claude_and_opencode_shapes() {
        let claude = r#"{"session_id":"s1","cwd":"/r","prompt":"hi","transcript_path":"/t"}"#;
        let ev: RawEvent = serde_json::from_str(claude).unwrap();
        assert_eq!(ev.prompt, "hi");
        assert_eq!(ev.session_id, "s1");

        let oc = r#"{"host":"opencode","session_id":"s2","cwd":"/r","prompt":"yo"}"#;
        let ev: RawEvent = serde_json::from_str(oc).unwrap();
        assert_eq!(ev.prompt, "yo");
        assert_eq!(ev.session_id, "s2");
    }

    #[test]
    fn strength_resolution() {
        assert_eq!(
            resolve_strength(Strength::Auto, Host::Claude),
            Strength::Soft
        );
        assert_eq!(
            resolve_strength(Strength::Auto, Host::Opencode),
            Strength::Hard
        );
        // Explicit settings pass through unchanged.
        assert_eq!(
            resolve_strength(Strength::Hard, Host::Claude),
            Strength::Hard
        );
    }

    #[test]
    fn select_threshold_and_cap() {
        let cfg = Config::default(); // min 0.30, margin 0.15, max 2
        let session = Session::default();
        let hits = vec![
            hit("a", 0.90, 0.0),
            hit("b", 0.85, 0.0),
            hit("c", 0.84, 0.0), // within margin but over the cap
            hit("d", 0.10, 0.0), // below threshold
        ];
        let got: Vec<String> = select(hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]); // capped at 2, d dropped
    }

    #[test]
    fn select_skips_loaded_and_denied() {
        let cfg = Config {
            deny: vec!["a".to_string()],
            ..Default::default()
        };
        let mut session = Session::default();
        session.mark("b", Source::Model);
        let hits = vec![
            hit("a", 0.90, 0.0),
            hit("b", 0.85, 0.0),
            hit("c", 0.80, 0.0),
        ];
        let got: Vec<String> = select(hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["c"]); // a denied, b already loaded, c within margin
    }

    #[test]
    fn select_margin_drops_weak_tail() {
        let cfg = Config::default(); // margin 0.15
        let session = Session::default();
        // Both clear the 0.30 floor, but b is far below the 0.90 leader.
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.50, 0.0)];
        let got: Vec<String> = select(hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a"]);
    }

    #[test]
    fn select_repeat_falls_silent() {
        // The strong match is already loaded; the rest are a weak tail measured
        // against the (still-global) leader, so nothing rides along.
        let cfg = Config::default();
        let mut session = Session::default();
        session.mark("a", Source::Ski);
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.50, 0.0)];
        assert!(select(hits, &cfg, &session).is_empty());
    }

    #[test]
    fn select_keeps_co_relevant_cluster() {
        let cfg = Config::default(); // margin 0.15
        let session = Session::default();
        let hits = vec![hit("a", 0.90, 0.0), hit("b", 0.80, 0.0)];
        let got: Vec<String> = select(hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["a", "b"]);
    }

    #[test]
    fn select_force_bypasses_threshold_on_keyword() {
        let cfg = Config {
            force: vec!["x".to_string()],
            ..Default::default()
        };
        let session = Session::default();
        // x is below threshold but forced with a keyword hit; y just below, no force.
        let hits = vec![hit("x", 0.1, 0.15), hit("y", 0.2, 0.0)];
        let got: Vec<String> = select(hits, &cfg, &session)
            .into_iter()
            .map(|h| h.id)
            .collect();
        assert_eq!(got, ["x"]);
    }

    #[test]
    fn render_claude_empty_is_silent() {
        assert_eq!(render_claude(&Decision::default()), "");
    }

    #[test]
    fn render_claude_wraps_context() {
        let d = Decision {
            inject: "ctx".to_string(),
            skills: vec!["a".to_string()],
        };
        let v: serde_json::Value = serde_json::from_str(&render_claude(&d)).unwrap();
        assert_eq!(v["hookSpecificOutput"]["hookEventName"], "UserPromptSubmit");
        assert_eq!(v["hookSpecificOutput"]["additionalContext"], "ctx");
    }

    #[test]
    fn render_opencode_always_json() {
        let v: serde_json::Value = serde_json::from_str(&render_opencode(&Decision::default()))
            .expect("opencode output is always valid JSON");
        assert_eq!(v["inject"], "");
        assert!(v["skills"].as_array().unwrap().is_empty());
    }
}
