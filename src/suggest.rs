//! `ski suggest` — turn the telemetry log into concrete, copy-pasteable tuning
//! actions. `ski history` *shows* the recall misses and false positives; this
//! closes the loop by saying what to do about them:
//!
//! - **Recall misses** (the model self-loaded a skill ski stayed silent on,
//!   repeatedly): suggest `force = ["<skill>"]` when one of the skill's existing
//!   keywords already appears in the missed prompts (so `force` would have fired
//!   as-is), and/or suggest new `keywords:` for its `SKILL.md` mined from the
//!   recurring content tokens of the missed prompts.
//! - **Repeat false positives** (ski injected it across several sessions and the
//!   model never once used it): suggest `deny = ["<skill>"]` — the config key the
//!   README calls the most-reached-for one.
//!
//! Everything here is *suggestion*, never mutation: ski does not edit the user's
//! config or SKILL.md files. Analysis is pure ([`analyze`]) and unit-testable
//! without IO; only [`run`] touches the filesystem. Read-only over the same JSONL
//! `ski history` reads, and equally tolerant of malformed lines.

use crate::history::{self, Verdict};
use crate::index::Index;
use crate::text::{match_tokens, norm_token};
use std::collections::{BTreeMap, BTreeSet};

/// Self-loads of a skill before a suggestion is emitted for it. One miss can be
/// a fluke; two of the same skill is a pattern worth acting on. Single-occurrence
/// misses are still listed (compactly) so a user watching a fresh log sees them
/// accumulate.
const MIN_MISS_EVIDENCE: usize = 2;

/// Sessions in which a skill was injected-and-never-used (with zero uses in *any*
/// session) before a `deny` is suggested. Deny is a blunt instrument — it silences
/// the skill entirely — so the bar is higher than for the recall-side suggestions,
/// in keeping with the project's err-toward-surfacing ethos.
const MIN_DENY_SESSIONS: u64 = 3;

/// How many mined keyword candidates to propose per skill.
const MAX_KEYWORDS: usize = 3;

/// How many sample prompts to keep per missed skill (for display).
const MAX_PROMPTS: usize = 3;

/// A skill the model keeps finding on its own while ski stays silent, with the
/// action(s) that would close the gap.
#[derive(Debug, PartialEq)]
pub struct Miss {
    pub skill: String,
    /// Total self-loads ski abstained or never surfaced on.
    pub occurrences: usize,
    /// Of those, how many ski had ranked near the top (near-miss) vs ranked deep
    /// (buried) vs never retrieved at all (absent) — tells the user whether the
    /// gap is the gate or the retrieval.
    pub near_miss: usize,
    pub buried: usize,
    pub absent: usize,
    /// Sample prompts the misses happened on (up to [`MAX_PROMPTS`]).
    pub prompts: Vec<String>,
    /// Whether one of the skill's *existing* keywords already appears in a missed
    /// prompt — i.e. `force = ["<skill>"]` would have fired with no other change.
    pub force_ready: bool,
    /// New keyword candidates mined from the missed prompts: content tokens
    /// recurring across them that the skill's keywords/description don't already
    /// carry. Empty when the misses share no vocabulary.
    pub keywords: Vec<String>,
}

/// A skill ski keeps injecting that the model never uses.
#[derive(Debug, PartialEq)]
pub struct Deny {
    pub skill: String,
    /// Sessions where it was injected and never used.
    pub fp_sessions: u64,
}

#[derive(Debug, Default, PartialEq)]
pub struct Suggestions {
    /// Recall-side actions, most-frequent miss first.
    pub misses: Vec<Miss>,
    /// Precision-side actions, most-frequent false positive first.
    pub denies: Vec<Deny>,
    /// Skills missed exactly once — no suggestion yet, listed so a pattern can be
    /// watched as the log grows.
    pub watch: Vec<String>,
}

/// Analyze a telemetry log against the (optional) index. The index supplies each
/// skill's existing keywords/description so suggestions don't propose what is
/// already there and can tell whether `force` is ready to fire; without it (no
/// index built yet) keyword mining still works, just unfiltered.
pub fn analyze(log: &str, idx: Option<&Index>) -> Suggestions {
    let mut out = Suggestions::default();

    // ---- Recall side: native picks ski abstained on or never surfaced. ----
    struct Acc {
        near: usize,
        buried: usize,
        absent: usize,
        prompts: Vec<String>,
    }
    let mut by_skill: BTreeMap<String, Acc> = BTreeMap::new();
    for row in history::compare(log) {
        let (near, buried, absent) = match row.verdict {
            Verdict::NearMiss { .. } => (1, 0, 0),
            Verdict::Buried { .. } => (0, 1, 0),
            Verdict::Absent => (0, 0, 1),
            // Agreed proves nothing to fix; NoRanking can't be placed.
            Verdict::Agreed | Verdict::NoRanking => continue,
        };
        let acc = by_skill.entry(row.native).or_insert(Acc {
            near: 0,
            buried: 0,
            absent: 0,
            prompts: Vec::new(),
        });
        acc.near += near;
        acc.buried += buried;
        acc.absent += absent;
        if !row.prompt.is_empty() && !acc.prompts.contains(&row.prompt) {
            acc.prompts.push(row.prompt);
        }
    }
    for (skill, acc) in by_skill {
        let occurrences = acc.near + acc.buried + acc.absent;
        if occurrences < MIN_MISS_EVIDENCE {
            out.watch.push(skill);
            continue;
        }
        let entry = idx.and_then(|i| i.get(&skill));
        // Tokens the skill already carries (keywords include its name tokens).
        let known: BTreeSet<String> = entry
            .map(|e| {
                let mut t: BTreeSet<String> = e
                    .keywords
                    .iter()
                    .flat_map(|k| match_tokens(k))
                    .map(|t| norm_token(&t))
                    .collect();
                t.extend(match_tokens(&e.description));
                t
            })
            .unwrap_or_default();
        // An existing keyword hitting any missed prompt means `force` fires as-is.
        let force_ready = entry.is_some_and(|e| {
            acc.prompts.iter().any(|p| {
                let toks: BTreeSet<String> =
                    match_tokens(p).iter().map(|t| norm_token(t)).collect();
                e.keywords.iter().any(|k| toks.contains(&norm_token(k)))
            })
        });
        let keywords = mine_keywords(&acc.prompts, &known);
        let mut prompts = acc.prompts;
        prompts.truncate(MAX_PROMPTS);
        out.misses.push(Miss {
            skill,
            occurrences,
            near_miss: acc.near,
            buried: acc.buried,
            absent: acc.absent,
            prompts,
            force_ready,
            keywords,
        });
    }
    out.misses.sort_by(|a, b| {
        b.occurrences
            .cmp(&a.occurrences)
            .then(a.skill.cmp(&b.skill))
    });

    // ---- Precision side: injected across sessions, never once used. ----
    let recd = history::recommended_by_session(log);
    let used = history::used_by_session(log);
    let mut fp_sessions: BTreeMap<String, u64> = BTreeMap::new();
    let mut ever_used: BTreeSet<&String> = BTreeSet::new();
    for (session, skills) in &recd {
        for skill in skills {
            if used.get(session).is_some_and(|u| u.contains(skill)) {
                ever_used.insert(skill);
            } else {
                *fp_sessions.entry(skill.clone()).or_default() += 1;
            }
        }
    }
    for skills in used.values() {
        ever_used.extend(skills.iter());
    }
    for (skill, n) in fp_sessions {
        if n >= MIN_DENY_SESSIONS && !ever_used.contains(&skill) {
            out.denies.push(Deny {
                skill,
                fp_sessions: n,
            });
        }
    }
    out.denies.sort_by(|a, b| {
        b.fp_sessions
            .cmp(&a.fp_sessions)
            .then(a.skill.cmp(&b.skill))
    });
    out
}

/// Keyword candidates from a skill's missed prompts: content tokens recurring in
/// at least two prompts (or all tokens when there is only one prompt would be too
/// noisy, so a single prompt yields nothing), minus what the skill already
/// carries. Most-recurrent first, capped at [`MAX_KEYWORDS`].
fn mine_keywords(prompts: &[String], known: &BTreeSet<String>) -> Vec<String> {
    if prompts.len() < 2 {
        return Vec::new();
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for p in prompts {
        let toks: BTreeSet<String> = match_tokens(p).into_iter().collect();
        for t in toks {
            *counts.entry(t).or_default() += 1;
        }
    }
    let mut cands: Vec<(String, usize)> = counts
        .into_iter()
        .filter(|(t, n)| *n >= 2 && !known.contains(t))
        .collect();
    cands.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    cands
        .into_iter()
        .take(MAX_KEYWORDS)
        .map(|(t, _)| t)
        .collect()
}

/// `ski suggest`: analyze the telemetry log for `host` and print actions.
pub fn run(host: crate::hook::Host) -> anyhow::Result<()> {
    let path = crate::paths::telemetry_path();
    let Ok(log) = std::fs::read_to_string(&path) else {
        println!(
            "no telemetry log at {} (enable with SKI_TELEMETRY=1, or telemetry = true in config.toml)",
            path.display()
        );
        return Ok(());
    };
    // Best-effort: suggestions degrade gracefully (unfiltered keyword mining, no
    // force-readiness check) when no index has been built yet.
    let idx = Index::load(&crate::paths::index_path(host)).ok().flatten();
    print_suggestions(&analyze(&log, idx.as_ref()));
    Ok(())
}

fn print_suggestions(s: &Suggestions) {
    if s.misses.is_empty() && s.denies.is_empty() && s.watch.is_empty() {
        println!("nothing to suggest: no repeated recall misses or unused injections in the log.");
        return;
    }
    if !s.misses.is_empty() {
        println!("recall misses — the model loaded these itself while ski stayed silent:\n");
        for m in &s.misses {
            println!(
                "  {}  ×{} self-loads (ski: near-miss ×{}, buried ×{}, never surfaced ×{})",
                m.skill, m.occurrences, m.near_miss, m.buried, m.absent
            );
            for p in &m.prompts {
                println!("    prompt: {}", crate::history::truncate(p, 100));
            }
            if m.force_ready {
                println!(
                    "    -> config.toml:  force = [\"{}\"]   (an existing keyword already hits these prompts)",
                    m.skill
                );
            }
            if !m.keywords.is_empty() {
                println!(
                    "    -> its SKILL.md: add keywords: [{}]{}",
                    m.keywords.join(", "),
                    if m.force_ready {
                        ""
                    } else {
                        "   (then force = [...] becomes effective too)"
                    }
                );
            }
            if !m.force_ready && m.keywords.is_empty() {
                println!(
                    "    -> the missed prompts share no vocabulary; consider expanding the skill's description"
                );
            }
            println!();
        }
    }
    if !s.denies.is_empty() {
        println!("repeat false positives — injected across sessions, never once used:\n");
        for d in &s.denies {
            println!("  {}  unused in {} sessions", d.skill, d.fp_sessions);
            println!("    -> config.toml:  deny = [\"{}\"]", d.skill);
            println!();
        }
    }
    if !s.watch.is_empty() {
        println!(
            "watching (one miss each, no suggestion yet): {}",
            s.watch.join(", ")
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::Entry;

    fn idx() -> Index {
        let entry = |id: &str, description: &str, keywords: &[&str]| Entry {
            id: id.to_string(),
            name: id.to_string(),
            description: description.to_string(),
            path: String::new(),
            keywords: keywords.iter().map(|k| k.to_string()).collect(),
            trigger_phrases: Vec::new(),
            hash: String::new(),
            embedding: Vec::new(),
        };
        Index {
            model: "test".into(),
            dim: 0,
            skills: vec![
                entry(
                    "uv-development",
                    "Manage Python projects with uv.",
                    &["uv", "python"],
                ),
                entry("pickup", "Resume a handoff.", &[]),
            ],
        }
    }

    // uv-development: self-loaded twice on prompts ski abstained on (one near-miss,
    // one absent). One of the prompts contains its existing keyword "uv" -> force
    // is ready. Both prompts share "dependency"/"lockfile" -> mined keywords.
    // pickup: injected in 3 sessions, never used -> deny candidate.
    // handoff: missed once -> watch list only.
    const LOG: &str = r#"
{"ts":1000,"kind":"recommend","session":"s1","stage":"rerank","prompt":"bump the dependency in the uv lockfile","considered":[{"id":"uv-development","score":-1.9}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":1100,"kind":"use","session":"s1","skill":"uv-development","via":"skill","prompt":"bump the dependency in the uv lockfile"}
{"ts":2000,"kind":"recommend","session":"s2","stage":"rerank","prompt":"pin that dependency in the lockfile","considered":[{"id":"other","score":-2.0}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":2100,"kind":"use","session":"s2","skill":"uv-development","via":"read","prompt":"pin that dependency in the lockfile"}
{"ts":3000,"kind":"recommend","session":"s3","stage":"cosine","prompt":"x","considered":[],"candidates":[{"id":"pickup","confidence":0.6}],"injected":[{"id":"pickup","confidence":0.6}]}
{"ts":4000,"kind":"recommend","session":"s4","stage":"cosine","prompt":"y","considered":[],"candidates":[{"id":"pickup","confidence":0.6}],"injected":[{"id":"pickup","confidence":0.6}]}
{"ts":5000,"kind":"recommend","session":"s5","stage":"cosine","prompt":"z","considered":[],"candidates":[{"id":"pickup","confidence":0.6}],"injected":[{"id":"pickup","confidence":0.6}]}
{"ts":6000,"kind":"recommend","session":"s6","stage":"rerank","prompt":"write the handoff notes","considered":[{"id":"handoff","score":-1.8}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":6100,"kind":"use","session":"s6","skill":"handoff","via":"skill","prompt":"write the handoff notes"}
"#;

    #[test]
    fn analyze_suggests_force_and_keywords_for_repeat_miss() {
        let s = analyze(LOG, Some(&idx()));
        assert_eq!(s.misses.len(), 1, "{s:?}");
        let m = &s.misses[0];
        assert_eq!(m.skill, "uv-development");
        assert_eq!(m.occurrences, 2);
        assert_eq!(m.near_miss, 1); // ranked #1 on the first prompt
        assert_eq!(m.absent, 1); // not in considered on the second
        assert!(m.force_ready); // "uv" keyword appears in the first prompt
                                // "dependency" and "lockfile" recur across both prompts and are not
                                // already carried by the skill; "uv"/"python" are known and excluded.
        assert!(m.keywords.contains(&"dependency".to_string()), "{m:?}");
        assert!(m.keywords.contains(&"lockfile".to_string()), "{m:?}");
        assert!(!m.keywords.contains(&"uv".to_string()));
    }

    #[test]
    fn analyze_suggests_deny_for_never_used_repeat_fp() {
        let s = analyze(LOG, Some(&idx()));
        assert_eq!(s.denies.len(), 1, "{s:?}");
        assert_eq!(s.denies[0].skill, "pickup");
        assert_eq!(s.denies[0].fp_sessions, 3);
    }

    #[test]
    fn single_miss_goes_to_watch_not_suggestion() {
        let s = analyze(LOG, Some(&idx()));
        assert_eq!(s.watch, vec!["handoff".to_string()]);
    }

    #[test]
    fn deny_requires_never_used_anywhere() {
        // Same 3 unused sessions, but one other session *did* use pickup: no deny.
        let log = format!(
            "{LOG}\n{}",
            r#"{"ts":7000,"kind":"use","session":"s7","skill":"pickup","via":"skill","prompt":"resume"}"#
        );
        let s = analyze(&log, Some(&idx()));
        assert!(s.denies.is_empty(), "{s:?}");
    }

    #[test]
    fn analyze_without_index_still_mines_keywords() {
        // No index: force-readiness can't be checked (false) and mining is
        // unfiltered ("uv" is no longer known, so it may appear as a candidate).
        let s = analyze(LOG, None);
        let m = &s.misses[0];
        assert!(!m.force_ready);
        assert!(m.keywords.contains(&"dependency".to_string()));
    }

    #[test]
    fn empty_log_yields_nothing() {
        assert_eq!(analyze("", Some(&idx())), Suggestions::default());
    }

    #[test]
    fn mine_keywords_needs_recurrence() {
        let known = BTreeSet::new();
        // A single prompt yields nothing (no recurrence signal).
        assert!(mine_keywords(&["one prompt only".to_string()], &known).is_empty());
        // Tokens appearing in both prompts survive; one-off tokens don't.
        let got = mine_keywords(
            &[
                "rotate the api credentials".to_string(),
                "rotate stale credentials".to_string(),
            ],
            &known,
        );
        assert!(got.contains(&"rotate".to_string()) && got.contains(&"credential".to_string()));
        assert!(!got.contains(&"stale".to_string()));
    }
}
