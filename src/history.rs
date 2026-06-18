//! `ski history` — aggregate the opt-in telemetry log into a calibration
//! readout, and `ski clear` — wipe per-session dedup state (and optionally the
//! log). Read-only over [`crate::telemetry`]'s JSONL; tolerant of malformed
//! lines (skips them) so a partially-written log still reports.

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

/// `ski history`: aggregate and print.
pub fn run() -> anyhow::Result<()> {
    let path = paths::telemetry_path();
    let Ok(log) = fs::read_to_string(&path) else {
        println!(
            "no telemetry log at {} (enable with SKI_TELEMETRY=1)",
            path.display()
        );
        return Ok(());
    };
    let s = aggregate(&log);
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
    Ok(())
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
}
