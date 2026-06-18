//! `ski history` — read the opt-in telemetry log. Two views over the same JSONL:
//! the default **aggregate** calibration readout (recommendations vs. actual
//! use), and a `--tail N` **per-event** listing that shows each recommendation's
//! prompt, stage, every candidate's confidence, which ids were injected, and
//! whether each injected skill was then used in that session — the view you want
//! when iterating on matcher quality. `ski clear` wipes per-session dedup state
//! (and optionally the log). All read-only and tolerant of malformed lines
//! (skips them) so a partially-written log still reports.

use crate::paths;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;

/// Parsed counts for the `history` readout.
#[derive(Debug, Default, PartialEq)]
pub struct Stats {
    pub recommend_events: u64,
    pub use_events: u64,
    pub sessions: usize,
    /// Distinct (session, skill) pairs that were recommended.
    pub recommended: u64,
    /// Recommended pairs the model then used in the same session.
    pub used_after_rec: u64,
    /// Recommended pairs never used (false positives).
    pub false_positives: u64,
    /// Used pairs never recommended in that session (recall misses).
    pub recall_misses: u64,
    /// skill id -> how many sessions recommended-but-unused.
    pub fp_by_skill: BTreeMap<String, u64>,
    /// skill id -> how many sessions used-but-unrecommended.
    pub miss_by_skill: BTreeMap<String, u64>,
}

/// One candidate skill with its shown confidence, for the per-event listing.
#[derive(Debug, PartialEq)]
pub struct Cand {
    pub id: String,
    pub confidence: f32,
}

/// A single `recommend` event, fully parsed for the `--tail` listing.
#[derive(Debug, PartialEq)]
pub struct RecEvent {
    pub ts: u128,
    pub session: String,
    pub stage: String,
    pub prompt: String,
    /// Every candidate that cleared the gate.
    pub candidates: Vec<Cand>,
    /// The subset that fit the char budget and was actually injected.
    pub injected: Vec<Cand>,
}

/// Parse every `recommend` event in log order (oldest first), keeping full
/// per-event detail. Pure; the listing view joins these against [`used_by_session`].
pub fn recommend_events(log: &str) -> Vec<RecEvent> {
    let mut out = Vec::new();
    for line in log.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        if v.get("kind").and_then(|k| k.as_str()) != Some("recommend") {
            continue;
        }
        out.push(RecEvent {
            ts: v.get("ts").and_then(|t| t.as_u64()).unwrap_or(0) as u128,
            session: str_field(&v, "session"),
            stage: str_field(&v, "stage"),
            prompt: str_field(&v, "prompt"),
            candidates: parse_cands(v.get("candidates")),
            injected: parse_cands(v.get("injected")),
        });
    }
    out
}

/// Per-session set of skill ids the model loaded itself (from `use` events), so
/// the listing can mark each injected candidate used vs. unused. Pure.
pub fn used_by_session(log: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut used: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        if v.get("kind").and_then(|k| k.as_str()) != Some("use") {
            continue;
        }
        if let Some(skill) = v.get("skill").and_then(|s| s.as_str()) {
            used.entry(str_field(&v, "session"))
                .or_default()
                .insert(skill.to_string());
        }
    }
    used
}

fn str_field(v: &serde_json::Value, key: &str) -> String {
    v.get(key)
        .and_then(|s| s.as_str())
        .unwrap_or("")
        .to_string()
}

fn parse_cands(v: Option<&serde_json::Value>) -> Vec<Cand> {
    v.and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let id = item.get("id")?.as_str()?.to_string();
                    let confidence = item
                        .get("confidence")
                        .and_then(|c| c.as_f64())
                        .unwrap_or(0.0) as f32;
                    Some(Cand { id, confidence })
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Aggregate raw JSONL log text. Pure, so it is unit-testable without IO.
pub fn aggregate(log: &str) -> Stats {
    // Per session: the set of recommended skill ids and the set of used skill ids.
    let mut rec: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut used: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut stats = Stats::default();

    for line in log.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
            continue;
        };
        let session = v.get("session").and_then(|s| s.as_str()).unwrap_or("");
        match v.get("kind").and_then(|k| k.as_str()) {
            Some("recommend") => {
                stats.recommend_events += 1;
                let entry = rec.entry(session.to_string()).or_default();
                if let Some(arr) = v.get("injected").and_then(|i| i.as_array()) {
                    for item in arr {
                        if let Some(id) = item.get("id").and_then(|i| i.as_str()) {
                            entry.insert(id.to_string());
                        }
                    }
                }
            }
            Some("use") => {
                stats.use_events += 1;
                if let Some(skill) = v.get("skill").and_then(|s| s.as_str()) {
                    used.entry(session.to_string())
                        .or_default()
                        .insert(skill.to_string());
                }
            }
            _ => {}
        }
    }

    let sessions: BTreeSet<&String> = rec.keys().chain(used.keys()).collect();
    stats.sessions = sessions.len();

    for session in sessions {
        let recommended = rec.get(session).cloned().unwrap_or_default();
        let consumed = used.get(session).cloned().unwrap_or_default();
        for id in &recommended {
            stats.recommended += 1;
            if consumed.contains(id) {
                stats.used_after_rec += 1;
            } else {
                stats.false_positives += 1;
                *stats.fp_by_skill.entry(id.clone()).or_default() += 1;
            }
        }
        for id in &consumed {
            if !recommended.contains(id) {
                stats.recall_misses += 1;
                *stats.miss_by_skill.entry(id.clone()).or_default() += 1;
            }
        }
    }
    stats
}

/// `ski history`: read the telemetry log. With `tail`, list that many recent
/// recommendation events individually; otherwise print the aggregate readout.
/// `session` filters the listing to sessions whose id contains the substring.
pub fn run(tail: Option<usize>, session: Option<&str>) -> anyhow::Result<()> {
    let path = paths::telemetry_path();
    let Ok(log) = fs::read_to_string(&path) else {
        println!(
            "no telemetry log at {} (enable with SKI_TELEMETRY=1)",
            path.display()
        );
        return Ok(());
    };
    match tail {
        Some(n) => print_events(&log, n, session),
        None => print_aggregate(&log),
    }
    Ok(())
}

/// Render the last `n` recommendation events with full per-call detail.
fn print_events(log: &str, n: usize, session_filter: Option<&str>) {
    let used = used_by_session(log);
    let mut events = recommend_events(log);
    if let Some(sf) = session_filter {
        events.retain(|e| e.session.contains(sf));
    }
    if events.is_empty() {
        println!("no recommendation events");
        return;
    }
    let total = events.len();
    let start = total.saturating_sub(n);
    println!("showing {} of {total} recommendation events", total - start);
    let now = now_ms();
    let empty = BTreeSet::new();
    for e in &events[start..] {
        let used_here = used.get(&e.session).unwrap_or(&empty);
        let injected_ids: BTreeSet<&str> = e.injected.iter().map(|c| c.id.as_str()).collect();
        println!(
            "\n{}  session {}  stage {}",
            ago(e.ts, now),
            short(&e.session),
            if e.stage.is_empty() { "?" } else { &e.stage },
        );
        println!("  prompt: {}", truncate(&e.prompt, 120));
        for c in &e.injected {
            let mark = if used_here.contains(&c.id) {
                "used"
            } else {
                "unused"
            };
            println!("  -> {:<26} {:.2}  {mark}", c.id, c.confidence);
        }
        // Candidates that cleared the gate but lost to the char budget.
        for c in &e.candidates {
            if !injected_ids.contains(c.id.as_str()) {
                println!("     {:<26} {:.2}  (over budget)", c.id, c.confidence);
            }
        }
    }
}

/// The aggregate calibration readout (the default `ski history` view).
fn print_aggregate(log: &str) {
    let s = aggregate(log);
    println!(
        "events: {} recommend, {} use across {} sessions",
        s.recommend_events, s.use_events, s.sessions
    );
    println!(
        "recommended: {}   used-after-rec: {} ({})   false positives: {} ({})",
        s.recommended,
        s.used_after_rec,
        pct(s.used_after_rec, s.recommended),
        s.false_positives,
        pct(s.false_positives, s.recommended),
    );
    println!(
        "recall misses (used, never recommended): {}",
        s.recall_misses
    );
    print_top("top false positives", &s.fp_by_skill);
    print_top("top recall misses", &s.miss_by_skill);
}

/// `ski clear`: wipe per-session dedup state; with `telemetry`, also the log.
pub fn clear(telemetry: bool) -> anyhow::Result<()> {
    let sessions = paths::sessions_dir();
    let removed = match fs::remove_dir_all(&sessions) {
        Ok(()) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e.into()),
    };
    println!(
        "{} session state at {}",
        if removed { "cleared" } else { "no" },
        sessions.display()
    );
    if telemetry {
        let log = paths::telemetry_path();
        match fs::remove_file(&log) {
            Ok(()) => println!("cleared telemetry log at {}", log.display()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
    }
    Ok(())
}

fn now_ms() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Human-readable age of an event timestamp. `?` for missing/future stamps.
fn ago(ts_ms: u128, now_ms: u128) -> String {
    if ts_ms == 0 || ts_ms > now_ms {
        return "?".to_string();
    }
    let secs = (now_ms - ts_ms) / 1000;
    if secs < 60 {
        format!("{secs}s ago")
    } else if secs < 3600 {
        format!("{}m ago", secs / 60)
    } else if secs < 86_400 {
        format!("{}h ago", secs / 3600)
    } else {
        format!("{}d ago", secs / 86_400)
    }
}

/// First 8 chars of a session id — enough to eyeball-group events.
fn short(session: &str) -> String {
    session.chars().take(8).collect()
}

/// One-line, length-capped prompt for the listing.
fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.chars().count() <= max {
        s
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

fn pct(n: u64, d: u64) -> String {
    if d == 0 {
        "0%".to_string()
    } else {
        format!("{:.0}%", 100.0 * n as f64 / d as f64)
    }
}

/// Print the highest-count skills, descending, capped at 8.
fn print_top(label: &str, by_skill: &BTreeMap<String, u64>) {
    if by_skill.is_empty() {
        return;
    }
    let mut rows: Vec<(&String, &u64)> = by_skill.iter().collect();
    rows.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let shown: Vec<String> = rows
        .iter()
        .take(8)
        .map(|(id, n)| format!("{id} ×{n}"))
        .collect();
    println!("{label}: {}", shown.join(", "));
}

#[cfg(test)]
mod tests {
    use super::*;

    const LOG: &str = r#"
{"kind":"recommend","session":"s1","injected":[{"id":"git-attribution","confidence":0.9},{"id":"pdf","confidence":0.6}]}
{"kind":"use","session":"s1","skill":"git-attribution","via":"skill"}
{"kind":"use","session":"s1","skill":"xlsx","via":"read"}
{"kind":"recommend","session":"s2","injected":[{"id":"pdf","confidence":0.7}]}
not json, should be skipped
{"kind":"recommend","session":"s2","injected":[{"id":"pdf","confidence":0.8}]}
"#;

    #[test]
    fn aggregate_counts_outcomes() {
        let s = aggregate(LOG);
        assert_eq!(s.recommend_events, 3);
        assert_eq!(s.use_events, 2);
        assert_eq!(s.sessions, 2);
        // s1 recommended git-attribution+pdf; s2 recommended pdf (deduped across 2 events).
        assert_eq!(s.recommended, 3);
        // git-attribution used in s1.
        assert_eq!(s.used_after_rec, 1);
        // pdf in s1 (unused), pdf in s2 (unused).
        assert_eq!(s.false_positives, 2);
        assert_eq!(s.fp_by_skill.get("pdf"), Some(&2));
        // xlsx used in s1 but never recommended -> recall miss.
        assert_eq!(s.recall_misses, 1);
        assert_eq!(s.miss_by_skill.get("xlsx"), Some(&1));
    }

    #[test]
    fn empty_log_is_zero() {
        assert_eq!(aggregate(""), Stats::default());
    }

    #[test]
    fn pct_guards_zero() {
        assert_eq!(pct(0, 0), "0%");
        assert_eq!(pct(1, 2), "50%");
    }

    const DETAIL_LOG: &str = r#"
{"ts":1000,"kind":"recommend","session":"sess-abcdef-1","stage":"cosine","prompt":"make a pdf","candidates":[{"id":"pdf","confidence":0.8},{"id":"docx","confidence":0.4}],"injected":[{"id":"pdf","confidence":0.8}]}
{"ts":2000,"kind":"use","session":"sess-abcdef-1","skill":"pdf","via":"skill"}
{"ts":3000,"kind":"recommend","session":"other-2","stage":"rerank","prompt":"line1\nline2","candidates":[{"id":"xlsx","confidence":0.5}],"injected":[{"id":"xlsx","confidence":0.5}]}
garbage
"#;

    #[test]
    fn recommend_events_parses_detail() {
        let evs = recommend_events(DETAIL_LOG);
        assert_eq!(evs.len(), 2);
        let first = &evs[0];
        assert_eq!(first.ts, 1000);
        assert_eq!(first.session, "sess-abcdef-1");
        assert_eq!(first.stage, "cosine");
        assert_eq!(first.prompt, "make a pdf");
        assert_eq!(first.candidates.len(), 2);
        assert_eq!(
            first.injected,
            vec![Cand {
                id: "pdf".into(),
                confidence: 0.8
            }]
        );
    }

    #[test]
    fn used_by_session_collects_use_events() {
        let used = used_by_session(DETAIL_LOG);
        assert!(used.get("sess-abcdef-1").unwrap().contains("pdf"));
        assert!(!used.contains_key("other-2"));
    }

    #[test]
    fn ago_buckets() {
        assert_eq!(ago(0, 10_000), "?");
        assert_eq!(ago(20_000, 10_000), "?"); // future
        assert_eq!(ago(9_000, 10_000), "1s ago");
        assert_eq!(ago(0, 120_000 + 1), "?");
        assert_eq!(ago(1_000, 121_000), "2m ago");
        assert_eq!(ago(1_000, 7_201_000), "2h ago");
        assert_eq!(ago(1_000, 172_801_000), "2d ago");
    }

    #[test]
    fn truncate_caps_and_flattens() {
        assert_eq!(truncate("a\nb", 10), "a b");
        assert_eq!(truncate("abcdef", 3), "abc…");
        assert_eq!(short("sess-abcdef-1"), "sess-abc");
    }
}
