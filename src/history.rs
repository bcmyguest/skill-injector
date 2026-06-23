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

/// One entry of a `recommend` event's pre-gate `considered` ranking: a skill and
/// the raw stage score it was ranked at (cosine-blend, reranker logit, or BM25).
#[derive(Debug, PartialEq)]
pub struct Ranked {
    pub id: String,
    pub score: f32,
}

/// A single `recommend` event, fully parsed for the `--tail` listing.
#[derive(Debug, PartialEq)]
pub struct RecEvent {
    pub ts: u128,
    pub session: String,
    pub stage: String,
    pub prompt: String,
    /// The pre-gate ranking ski considered (top-K, id + raw stage score). Present
    /// on every prompt, including abstentions — the field the `--compare` view
    /// joins a native pick against. Empty for events logged before this field.
    pub considered: Vec<Ranked>,
    /// Every candidate that cleared the gate.
    pub candidates: Vec<Cand>,
    /// The subset that fit the char budget and was actually injected.
    pub injected: Vec<Cand>,
    /// Why nothing was injected (`below_gate`, `empty_text`), or `None` on a
    /// successful inject.
    pub abstained: Option<String>,
}

/// A single `use` event (the model loaded a skill itself), for the `--tail`
/// listing. `prompt` is the call that was active when it loaded, if telemetry
/// captured it (empty otherwise).
#[derive(Debug, PartialEq)]
pub struct UseEvent {
    pub ts: u128,
    pub session: String,
    pub skill: String,
    pub via: String,
    pub prompt: String,
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
            considered: parse_ranked(v.get("considered")),
            candidates: parse_cands(v.get("candidates")),
            injected: parse_cands(v.get("injected")),
            abstained: v
                .get("abstained")
                .and_then(|a| a.as_str())
                .map(str::to_string),
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

/// Parse every `use` event in log order (oldest first), keeping the prompt for
/// the listing. Pure.
pub fn use_events(log: &str) -> Vec<UseEvent> {
    let mut out = Vec::new();
    for line in log.lines() {
        let Ok(v) = serde_json::from_str::<serde_json::Value>(line.trim()) else {
            continue;
        };
        if v.get("kind").and_then(|k| k.as_str()) != Some("use") {
            continue;
        }
        let Some(skill) = v.get("skill").and_then(|s| s.as_str()) else {
            continue;
        };
        out.push(UseEvent {
            ts: v.get("ts").and_then(|t| t.as_u64()).unwrap_or(0) as u128,
            session: str_field(&v, "session"),
            skill: skill.to_string(),
            via: str_field(&v, "via"),
            prompt: str_field(&v, "prompt"),
        });
    }
    out
}

/// Per-session set of skill ids that were injected (from `recommend` events), so
/// the listing can label a `use` as acting on a recommendation vs. a recall miss.
/// Pure.
pub fn recommended_by_session(log: &str) -> BTreeMap<String, BTreeSet<String>> {
    let mut rec: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for e in recommend_events(log) {
        let entry = rec.entry(e.session).or_default();
        for c in e.injected {
            entry.insert(c.id);
        }
    }
    rec
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

fn parse_ranked(v: Option<&serde_json::Value>) -> Vec<Ranked> {
    v.and_then(|a| a.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let id = item.get("id")?.as_str()?.to_string();
                    let score = item.get("score").and_then(|s| s.as_f64()).unwrap_or(0.0) as f32;
                    Some(Ranked { id, score })
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

/// Rank at/under which an abstained-on native pick counts as a near-miss (a
/// tunable edge) rather than buried. ski's top few are the realistically
/// reachable band for a looser gate or a stronger retrieval channel.
const NEAR_MISS_RANK: usize = 3;

/// How ski's ranking lines up with the native chooser's pick on one prompt.
#[derive(Debug, PartialEq)]
pub enum Verdict {
    /// ski injected the same skill the model then used. Redundant — and because
    /// the model saw ski's nudge first, this does *not* prove ski caused the pick.
    Agreed,
    /// ski abstained but ranked the native pick in its top [`NEAR_MISS_RANK`] —
    /// the tunable edge: a looser gate or stronger channel could have surfaced it.
    NearMiss { rank: usize, score: f32 },
    /// ski abstained and ranked the native pick deeper in its considered top-K.
    Buried { rank: usize, score: f32 },
    /// ski never surfaced the native pick in its considered top-K — the retrieval
    /// ceiling; narrowing the gate cannot win this one.
    Absent,
    /// No considered ranking was logged for the prompt (telemetry was off then, or
    /// a pre-feature event), so the pick can't be placed.
    NoRanking,
}

/// One native-chooser pick (a `use` event) joined to ski's ranking on the same
/// prompt (the matching `recommend` event).
#[derive(Debug, PartialEq)]
pub struct CompareRow {
    pub session: String,
    pub prompt: String,
    pub stage: String,
    /// The skill the native chooser picked.
    pub native: String,
    pub via: String,
    pub verdict: Verdict,
}

/// Join every native-chooser pick to where ski ranked it on the same prompt.
/// Pure, so the classification is unit-testable without IO. A `use` is matched to
/// the latest `recommend` event with the same session + prompt at/just before it.
pub fn compare(log: &str) -> Vec<CompareRow> {
    let recs = recommend_events(log);
    use_events(log)
        .into_iter()
        .map(|u| {
            let rec = match_recommend(&recs, &u);
            let (stage, verdict) = match rec {
                None => (String::new(), Verdict::NoRanking),
                Some(r) => (r.stage.clone(), classify(r, &u.skill)),
            };
            CompareRow {
                session: u.session,
                prompt: u.prompt,
                stage,
                native: u.skill,
                via: u.via,
                verdict,
            }
        })
        .collect()
}

/// The `recommend` event a `use` should be scored against: same session + prompt,
/// preferring the latest one at or before the use (the call that was active), and
/// falling back to the earliest match when none precede it (clock skew / ordering).
fn match_recommend<'a>(recs: &'a [RecEvent], u: &UseEvent) -> Option<&'a RecEvent> {
    if u.prompt.is_empty() {
        return None;
    }
    let mut matches: Vec<&RecEvent> = recs
        .iter()
        .filter(|r| r.session == u.session && r.prompt == u.prompt)
        .collect();
    if matches.is_empty() {
        return None;
    }
    matches.sort_by_key(|r| r.ts);
    matches
        .iter()
        .rev()
        .find(|r| r.ts <= u.ts)
        .or_else(|| matches.first())
        .copied()
}

/// Place `native` (the picked skill) within one recommend event's outcome.
fn classify(r: &RecEvent, native: &str) -> Verdict {
    if r.injected.iter().any(|c| c.id == native) {
        return Verdict::Agreed;
    }
    match r.considered.iter().position(|c| c.id == native) {
        Some(i) => {
            let rank = i + 1;
            let score = r.considered[i].score;
            if rank <= NEAR_MISS_RANK {
                Verdict::NearMiss { rank, score }
            } else {
                Verdict::Buried { rank, score }
            }
        }
        None => Verdict::Absent,
    }
}

/// `ski history`: read the telemetry log. With `tail`, list that many recent
/// recommendation events individually; with `compare`, show ski's pick vs the
/// native chooser's pick per prompt; otherwise print the aggregate readout.
/// `session` filters the listing to sessions whose id contains the substring.
pub fn run(tail: Option<usize>, session: Option<&str>, compare_view: bool) -> anyhow::Result<()> {
    let path = paths::telemetry_path();
    let Ok(log) = fs::read_to_string(&path) else {
        println!(
            "no telemetry log at {} (enable with SKI_TELEMETRY=1)",
            path.display()
        );
        return Ok(());
    };
    if compare_view {
        print_compare(&log, session);
    } else {
        match tail {
            Some(n) => print_events(&log, n, session),
            None => print_aggregate(&log),
        }
    }
    Ok(())
}

/// One entry in the merged per-call timeline: a recommendation we made, or a
/// skill the model loaded itself.
enum Ev<'a> {
    Rec(&'a RecEvent),
    Use(&'a UseEvent),
}

impl Ev<'_> {
    fn ts(&self) -> u128 {
        match self {
            Ev::Rec(e) => e.ts,
            Ev::Use(u) => u.ts,
        }
    }
}

/// Render the last `n` events (recommendations and self-loads, interleaved by
/// time) with full per-call detail.
fn print_events(log: &str, n: usize, session_filter: Option<&str>) {
    let used = used_by_session(log);
    let recd = recommended_by_session(log);
    let recs = recommend_events(log);
    let uses = use_events(log);
    let keep = |s: &str| session_filter.is_none_or(|sf| s.contains(sf));

    let mut timeline: Vec<Ev> = recs
        .iter()
        .filter(|e| keep(&e.session))
        .map(Ev::Rec)
        .chain(uses.iter().filter(|u| keep(&u.session)).map(Ev::Use))
        .collect();
    // Stable sort keeps a recommend before a same-millisecond use of it.
    timeline.sort_by_key(Ev::ts);

    if timeline.is_empty() {
        println!("no events");
        return;
    }
    let total = timeline.len();
    let start = total.saturating_sub(n);
    println!("showing {} of {total} events", total - start);
    let now = now_ms();
    let empty = BTreeSet::new();
    for ev in &timeline[start..] {
        match ev {
            Ev::Rec(e) => {
                let used_here = used.get(&e.session).unwrap_or(&empty);
                let injected_ids: BTreeSet<&str> =
                    e.injected.iter().map(|c| c.id.as_str()).collect();
                println!(
                    "\n{}  session {}  rec  stage {}",
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
            Ev::Use(u) => {
                let acted = recd.get(&u.session).is_some_and(|s| s.contains(&u.skill));
                let tag = if acted { "acted on rec" } else { "RECALL MISS" };
                println!(
                    "\n{}  session {}  use  {} via {} ({tag})",
                    ago(u.ts, now),
                    short(&u.session),
                    u.skill,
                    u.via,
                );
                if !u.prompt.is_empty() {
                    println!("  prompt: {}", truncate(&u.prompt, 120));
                }
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

/// The `--compare` view: every native-chooser pick joined to where ski ranked it.
/// The whole point is the NEAR-MISS bucket — prompts the model found a skill on
/// that ski ranked but gated out — versus ABSENT, where ski never surfaced it.
fn print_compare(log: &str, session_filter: Option<&str>) {
    let rows: Vec<CompareRow> = compare(log)
        .into_iter()
        .filter(|r| session_filter.is_none_or(|sf| r.session.contains(sf)))
        .collect();
    if rows.is_empty() {
        println!("no native-chooser picks logged (need `use` events; enable SKI_TELEMETRY=1)");
        return;
    }
    let sessions: BTreeSet<&str> = rows.iter().map(|r| r.session.as_str()).collect();
    let (mut agreed, mut near, mut buried, mut absent, mut no_rank) = (0, 0, 0, 0, 0);
    for r in &rows {
        match r.verdict {
            Verdict::Agreed => agreed += 1,
            Verdict::NearMiss { .. } => near += 1,
            Verdict::Buried { .. } => buried += 1,
            Verdict::Absent => absent += 1,
            Verdict::NoRanking => no_rank += 1,
        }
    }
    println!(
        "ski vs native chooser — {} picks across {} sessions",
        rows.len(),
        sessions.len()
    );
    println!("  agreed (ski injected it too):                {agreed}");
    println!("  NEAR-MISS (ski ranked it ≤{NEAR_MISS_RANK}, abstained): {near}   <- tunable edge");
    println!("  buried (ranked deeper, abstained):           {buried}");
    println!("  absent (ski never surfaced it):              {absent}   <- retrieval ceiling");
    if no_rank > 0 {
        println!("  no ranking logged for the prompt:            {no_rank}");
    }
    println!(
        "\nnote: native picks are observed *after* ski injects, so \"agreed\" doesn't prove ski\n\
         caused it. The clean edge signal is NEAR-MISS/buried — the model found a skill itself\n\
         that ski ranked but gated out."
    );

    // Edge candidates: native picked it, ski ranked but abstained. Best rank first.
    let mut edges: Vec<(usize, f32, &CompareRow)> = rows
        .iter()
        .filter_map(|r| match r.verdict {
            Verdict::NearMiss { rank, score } | Verdict::Buried { rank, score } => {
                Some((rank, score, r))
            }
            _ => None,
        })
        .collect();
    edges.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.prompt.cmp(&b.2.prompt)));
    if !edges.is_empty() {
        println!("\nedge candidates (native picked it, ski abstained):");
        for (rank, score, r) in &edges {
            println!(
                "  {:<60} -> {} via {}  ski #{rank} score {score:.3}  [{}]",
                truncate(&r.prompt, 60),
                r.native,
                r.via,
                if r.stage.is_empty() { "?" } else { &r.stage },
            );
        }
    }

    let absent_rows: Vec<&CompareRow> = rows
        .iter()
        .filter(|r| r.verdict == Verdict::Absent)
        .collect();
    if !absent_rows.is_empty() {
        println!("\nabsent (ski never surfaced — retrieval miss):");
        for r in &absent_rows {
            println!(
                "  {:<60} -> {} via {}",
                truncate(&r.prompt, 60),
                r.native,
                r.via
            );
        }
    }
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
    fn use_events_parse_prompt_when_present() {
        let log = r#"
{"ts":5000,"kind":"use","session":"s1","skill":"xlsx","via":"read","prompt":"clean this csv"}
{"ts":6000,"kind":"use","session":"s1","skill":"pdf","via":"skill"}
"#;
        let evs = use_events(log);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].skill, "xlsx");
        assert_eq!(evs[0].via, "read");
        assert_eq!(evs[0].prompt, "clean this csv");
        assert_eq!(evs[1].prompt, ""); // no prompt field -> empty
    }

    #[test]
    fn recommended_by_session_uses_injected_ids() {
        let recd = recommended_by_session(DETAIL_LOG);
        assert!(recd.get("sess-abcdef-1").unwrap().contains("pdf"));
        // docx cleared the gate but only pdf was injected in this fixture.
        assert!(!recd.get("sess-abcdef-1").unwrap().contains("docx"));
        assert!(recd.get("other-2").unwrap().contains("xlsx"));
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

    #[test]
    fn recommend_events_parse_considered_and_abstained() {
        let log = r#"
{"ts":1,"kind":"recommend","session":"s","stage":"rerank","prompt":"p","considered":[{"id":"xlsx","score":-1.96},{"id":"pdf","score":-2.1}],"candidates":[],"injected":[],"abstained":"below_gate"}
"#;
        let evs = recommend_events(log);
        assert_eq!(evs.len(), 1);
        assert_eq!(
            evs[0].considered,
            vec![
                Ranked {
                    id: "xlsx".into(),
                    score: -1.96
                },
                Ranked {
                    id: "pdf".into(),
                    score: -2.1
                },
            ]
        );
        assert_eq!(evs[0].abstained.as_deref(), Some("below_gate"));
        // A legacy event without the new fields parses to empty/None, not an error.
        let legacy =
            r#"{"kind":"recommend","session":"s","injected":[{"id":"pdf","confidence":0.7}]}"#;
        let ev = &recommend_events(legacy)[0];
        assert!(ev.considered.is_empty());
        assert_eq!(ev.abstained, None);
    }

    // One recommend + one use per session, exercising each verdict.
    const COMPARE_LOG: &str = r#"
{"ts":1000,"kind":"recommend","session":"s1","stage":"rerank","prompt":"make a chart","considered":[{"id":"xlsx","score":-1.9},{"id":"pdf","score":-2.1}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":1100,"kind":"use","session":"s1","skill":"xlsx","via":"skill","prompt":"make a chart"}
{"ts":2000,"kind":"recommend","session":"s2","stage":"cosine","prompt":"set up python","considered":[{"id":"uv-setup","score":0.7}],"candidates":[{"id":"uv-setup","confidence":0.7}],"injected":[{"id":"uv-setup","confidence":0.7}]}
{"ts":2100,"kind":"use","session":"s2","skill":"uv-setup","via":"skill","prompt":"set up python"}
{"ts":3000,"kind":"recommend","session":"s3","stage":"rerank","prompt":"deep","considered":[{"id":"a","score":0.1},{"id":"b","score":0.1},{"id":"c","score":0.1},{"id":"d","score":0.1},{"id":"gold","score":0.0}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":3100,"kind":"use","session":"s3","skill":"gold","via":"read","prompt":"deep"}
{"ts":4000,"kind":"recommend","session":"s4","stage":"cosine","prompt":"weird","considered":[{"id":"x","score":0.2}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":4100,"kind":"use","session":"s4","skill":"notranked","via":"skill","prompt":"weird"}
{"ts":5100,"kind":"use","session":"s5","skill":"orphan","via":"skill","prompt":"no rec here"}
"#;

    #[test]
    fn compare_classifies_each_verdict() {
        let rows = compare(COMPARE_LOG);
        let by: std::collections::HashMap<&str, &Verdict> = rows
            .iter()
            .map(|r| (r.native.as_str(), &r.verdict))
            .collect();
        assert_eq!(
            by["xlsx"],
            &Verdict::NearMiss {
                rank: 1,
                score: -1.9
            }
        );
        assert_eq!(by["uv-setup"], &Verdict::Agreed);
        assert_eq!(
            by["gold"],
            &Verdict::Buried {
                rank: 5,
                score: 0.0
            }
        );
        assert_eq!(by["notranked"], &Verdict::Absent);
        // A use with no matching recommend (or no prompt) can't be placed.
        assert_eq!(by["orphan"], &Verdict::NoRanking);
    }

    #[test]
    fn classify_near_miss_rank_boundary() {
        // rank == NEAR_MISS_RANK is still a near-miss; one deeper is buried.
        let log = r#"
{"ts":1,"kind":"recommend","session":"s","stage":"rerank","prompt":"p","considered":[{"id":"a","score":0.3},{"id":"b","score":0.2},{"id":"gold","score":0.1},{"id":"d","score":0.0}],"candidates":[],"injected":[],"abstained":"below_gate"}
"#;
        let ev = &recommend_events(log)[0];
        assert_eq!(
            classify(ev, "gold"),
            Verdict::NearMiss {
                rank: 3,
                score: 0.1
            }
        );
        assert_eq!(
            classify(ev, "d"),
            Verdict::Buried {
                rank: 4,
                score: 0.0
            }
        );
    }

    #[test]
    fn match_recommend_prefers_latest_at_or_before_use() {
        // Same prompt ranked twice; the use should bind to the one active at its time.
        let log = r#"
{"ts":1000,"kind":"recommend","session":"s","stage":"cosine","prompt":"p","considered":[{"id":"a","score":0.5}],"candidates":[],"injected":[],"abstained":"below_gate"}
{"ts":3000,"kind":"recommend","session":"s","stage":"cosine","prompt":"p","considered":[{"id":"a","score":0.9}],"candidates":[],"injected":[],"abstained":"below_gate"}
"#;
        let recs = recommend_events(log);
        let u = UseEvent {
            ts: 2000,
            session: "s".into(),
            skill: "a".into(),
            via: "skill".into(),
            prompt: "p".into(),
        };
        // ts 2000 binds to the 1000 event (latest at/before), not the later 3000 one.
        assert_eq!(match_recommend(&recs, &u).unwrap().ts, 1000);
    }
}
